from parlai.agents.hugging_face.gpt2 import Gpt2Agent

import torch
import torch.nn as nn
import torch.nn.functional as F
from parlai.utils.torch import padded_tensor
from parlai.core.metrics import AverageMetric

from parlai.agents.knowledge_grounded_generator.kg_utils import NOCONCEPT_TOKEN, NORELATION_TOKEN, ConceptGraph
from parlai.agents.knowledge_grounded_generator.kg_model import KnowledgeGroundedModel
import parlai.utils.logging as logging

class KG_loss(nn.Module):

    def __init__(self, ignore_index, invalid, alpha, beta):
        super().__init__()
        self.ignore_index = ignore_index
        self.invalid = invalid
        self.alpha = alpha
        self.beta = beta

    def forward(self, lm_probs, labels, triple_prob, triple_labels, gate, gate_labels):

        # Compute overall loss
        gen_loss_fn = nn.NLLLoss(ignore_index=self.ignore_index)
        probs_clamp = lm_probs.clamp(min=1e-5)
        gen_loss = gen_loss_fn(probs_clamp.log().view(-1, lm_probs.size(-1)), labels.view(-1))

        # Compute and record triple loss
        triple_mask = (triple_labels != self.invalid).unsqueeze(1).expand_as(triple_prob).float()
        triple_labels = triple_labels.unsqueeze(1).expand_as(triple_prob) * triple_mask
        triple_loss_fn = nn.BCELoss(weight=triple_mask.view(-1), reduction='mean')
        triple_loss = triple_loss_fn(triple_prob.view(-1), triple_labels.view(-1).float())

        # Compute and record gate loss   
        gate_mask = (gate_labels != self.invalid).float()
        gate_labels.masked_fill_((gate_labels == self.invalid), 0)
        lm_mask = (gate_labels.sum(1) != 0).float().unsqueeze(1)
        gate_mask = lm_mask.expand_as(gate_labels) * gate_mask
        gate_loss_fn = nn.BCELoss(weight=gate_mask.view(-1), reduction='mean')
        gate_loss = gate_loss_fn(gate.view(-1), gate_labels.view(-1).float())

        combined_loss = gen_loss + self.alpha * gate_loss + self.beta * triple_loss

        return combined_loss, gen_loss, triple_loss, gate_loss   


class KnowledgeGroundedGeneratorAgent(Gpt2Agent):

    @classmethod
    def add_cmdline_args(cls, parser, partial_opt=None):
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        """
        Add CLI arguments.
        """
        # Make sure to add all of TorchGeneratorAgent's arguments
        super().add_cmdline_args(parser, partial_opt=partial_opt)

        # Add custom arguments only for this model.
        group = parser.add_argument_group('KG Generator Agent')
        group.add_argument(
            '--kg-datadir', 
            type=str, 
            default='/users/FrankVerhoef/Programming/Project_AI/ParlAI/data/kg_data/', 
            help='dir for knowledge graph data'
        )
        group.add_argument(
            '--concepts', 
            type=str, 
            default='concept.txt', 
            help='file with concept vocab'
        )
        group.add_argument(
            '--relations', 
            type=str, 
            default='relation.txt', 
            help='file with relation names'
        )
        group.add_argument(
            '--dataset-concepts', 
            type=str, 
            default='total_concepts.txt', 
            help='file with dataset concepts'
        )
        group.add_argument(
            '--kg', 
            type=str, 
            default='cpnet_25.graph', 
            help='file with knowledge graph'
        )
        group.add_argument(
            '-hid', '--hidden-size', type=int, default=256, help='Hidden size.'
        )
        group.add_argument(
            '-emb', '--embedding-size', type=int, default=768, help='Hidden size.'
        )
        group.add_argument(
            "--source-length",
            type=int,
            default=16,
            help="Length of input sentence."
        )
        group.add_argument(
            "--num-hops",
            type=int,
            default=2,
            help="Number of hops in the graph to look for related concepts."
        )
        group.add_argument(
            "--max-concepts",
            type=int,
            default=400,
            help="Maximum number of related concepts to include."
        )
        group.add_argument(
            "--max-triples",
            type=int,
            default=800,
            help="Maximum number of relations to include."
        )
        group.add_argument(
            "--alpha",
            type=float,
            default=1.0,
            help="Parameter for impact of gate loss in loss calculation."
        )
        group.add_argument(
            "--beta",
            type=float,
            default=1.0,
            help="Parameter for impact of triple loss in loss calculation."
        )
        group.add_argument(
            "--gamma",
            type=float,
            default=0.8,
            help="Parameter for calculation of probabilities of triple heads"
        )
        group.add_argument(
            "--gate",
            type=float,
            default=None,
            help="If set, uses a fixed gate probability [0.0 - 1.0]"
        )
        group.add_argument(
            "--aggregate-method",
            type=str,
            default="max",
            choices=["avg", "max"],
            help="How to aggregate probabilities on graph nodes."
        )
        return parser


    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        self.model_vocab = self.dict.keys()
        self.num_hops = opt['num_hops']
        self.max_concepts = opt['max_concepts']
        self.max_triples = opt['max_triples']
        self.kg = ConceptGraph(opt['kg_datadir'], opt['concepts'], opt['relations'], opt['dataset_concepts'], opt['kg'])
        self.vocab_map, self.map_mask = self.build_vocab_map()
        logging.info("Initialized KnowledgeGroundedGeneratorAgent")


    def build_model(self):

        model = KnowledgeGroundedModel(self.opt, self.dict)
        return model


    def build_vocab_map(self):

        vocab_map, map_mask = [], []
        for idx in self.model_vocab:
            try: 
                pos = self.kg.concept2id.index(idx)
                vocab_map.append(pos)
                map_mask.append(1)
            except:
                vocab_map.append(0)
                map_mask.append(0)
        assert(len(vocab_map) == len(self.model_vocab))

        return vocab_map, map_mask


    def observe(self, observation):
        logging.debug('=== Observation ===\n{}'.format(observation['text']))
        observation = super().observe(observation)

        # Match with concepts in knowledge graph
        labels = observation['labels'] if 'labels' in observation.keys() else observation['eval_labels']
        concepts = self.kg.match_mentioned_concepts(observation['text'], ' '.join(labels))
        # for k, v in concepts.items():
        #     print(k, v)
        related_concepts = self.kg.find_neighbours(concepts['qc'], concepts['ac'], num_hops=self.num_hops, max_B=100)[0]
        # print("Related concepts: ")
        # for k, v in related_concepts.items():
        #     print(k, v[:10])
        
        filtered_data = self.kg.filter_directed_triple(related_concepts, max_concepts=self.max_concepts, max_triples=self.max_triples)

        # Construct list with gate_labels
        target_concept_ids = [self.dict.txt2vec(' ' + c)[0] for c in concepts['ac']]
        label_ids = self.dict.txt2vec(labels[0])
        gate_labels = [1 if x in target_concept_ids else 0 for x in label_ids] + [0] # add 0 for end-token

        # TODO: Think of alternative way to match concepts to tokens
        # TODO: Truncate or pad to max number of concepts and relations

        # Info about the related concepts
        observation['related_concepts'] = filtered_data['concepts']
        observation['concept_token_ids'] = torch.LongTensor([self.dict.txt2vec(' ' + c)[0] for c in filtered_data['concepts']])
        observation['concept_labels'] = torch.LongTensor(filtered_data['labels'])
        observation['distances'] = torch.LongTensor(filtered_data['distances'])
        observation['gate_labels'] = torch.LongTensor(gate_labels)

        # Info about relations to related concepts
        observation['relation_ids'] = torch.LongTensor(filtered_data['relation_ids'])
        observation['head_idx'] = torch.LongTensor(filtered_data['head_idx'])
        observation['tail_idx'] = torch.LongTensor(filtered_data['tail_idx'])
        observation['triple_labels'] = torch.LongTensor(filtered_data['triple_labels'])

        logging.debug("Found {} related concepts and {} relations".format(
            len(observation['related_concepts']), len(observation['relation_ids']))
        )
        logging.debug("\t{}".format(', '.join(observation['related_concepts'][:10])))

        return observation


    def act(self):

        logging.debug("=== Action === ")
        reply = super().act()

        return reply

    def batchify(self, obs_batch, sort=False):
        batch = super().batchify(obs_batch, sort=sort)

        batch['concept_ids'], _ = padded_tensor(
            [obs_batch[i]['concept_token_ids'] for i in batch.valid_indices],
            pad_idx = self.kg.concept2id[NOCONCEPT_TOKEN]            
        )
        batch['concept_labels'], _ = padded_tensor(
            [obs_batch[i]['concept_labels'] for i in batch.valid_indices],
            pad_idx = self.kg.concept2id[NOCONCEPT_TOKEN]            
        )
        batch['distances'], _ = padded_tensor(
            [obs_batch[i]['distances'] for i in batch.valid_indices],
            pad_idx=0
        )
        batch['relation_ids'], _ = padded_tensor(
            [obs_batch[i]['relation_ids'] for i in batch.valid_indices], 
            pad_idx = self.kg.relation2id[NORELATION_TOKEN]
        )
        batch['head_idx'], _ = padded_tensor(
            [obs_batch[i]['head_idx'] for i in batch.valid_indices],
            pad_idx=0
        )
        batch['tail_idx'], _ = padded_tensor(
            [obs_batch[i]['tail_idx'] for i in batch.valid_indices],
            pad_idx=0
        )
        batch['triple_labels'], _ = padded_tensor(
            [obs_batch[i]['triple_labels'] for i in batch.valid_indices],
            pad_idx=-1
        )
        batch['gate_labels'], _ = padded_tensor(
            [obs_batch[i]['gate_labels'] for i in batch.valid_indices],
            pad_idx=-1
        )
        batch['vocab_map'] = torch.LongTensor(self.vocab_map)
        batch['map_mask'] = torch.LongTensor(self.map_mask)

        return batch


    def _encoder_input(self, batch):
        return self._model_input(batch)


    def _model_input(self, batch):
        return (
            batch.text_vec,
            batch.concept_ids,
            batch.distances,
            batch.relation_ids,
            batch.head_idx,
            batch.tail_idx,
            batch.triple_labels,
            batch.gate_labels,
            batch.vocab_map,
            batch.map_mask
        )


    def _get_initial_decoder_input(self, bsz, beam_size, dev):
        return (
            torch.LongTensor([self.dict.start_idx])
            .expand(bsz * beam_size, 1)
            .to(dev)
        )


    def compute_loss(self, batch, return_output=False):
        """
        Compute and return the loss for the given batch.
        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')

        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        scores, preds, encoder_states, triple_prob, gate = model_output

        combined_loss, gen_loss, triple_loss, gate_loss = self.criterion(
            scores, batch.label_vec, 
            triple_prob, batch.triple_labels, 
            gate, batch.gate_labels
        )

        self.record_local_metric('loss', AverageMetric.many([combined_loss.item()]))
        self.record_local_metric('loss_gen', AverageMetric.many([gen_loss.item()]))
        self.record_local_metric('loss_triple', AverageMetric.many([triple_loss.item()]))
        self.record_local_metric('loss_gate', AverageMetric.many([gate_loss.item()]))

         # actually do backwards loss
        loss = combined_loss.sum()
        # loss /= target_tokens.sum()  # average loss per token

        if return_output:
            return (loss, model_output)
        else:
            return loss


    def build_criterion(self):
        """
        Construct and return the loss function.
        Uses parameters alpha and beta to determine how much of gate loss (x alpha) and triple loss (x beta)
        is added to the final loss
        """
        return KG_loss(ignore_index=self.NULL_IDX, invalid=-1, alpha = self.opt['alpha'], beta = self.opt['beta'])
