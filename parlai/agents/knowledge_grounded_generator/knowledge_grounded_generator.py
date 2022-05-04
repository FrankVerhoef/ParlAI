from parlai.agents.hugging_face.gpt2 import Gpt2Agent
from parlai.core.torch_agent import Output

import torch
import torch.nn as nn
import torch.nn.functional as F
from parlai.core.torch_generator_agent import TorchGeneratorAgent

from parlai.agents.knowledge_grounded_generator.kg_utils import ConceptGraph, filter_directed_triple
from parlai.agents.knowledge_grounded_generator.multihop import KnowledgeGroundedModel
import parlai.utils.logging as logging

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
            '--concepts', 
            type=str, 
            default='/users/FrankVerhoef/Programming/Project_AI/multigen/data/concept.txt', 
            help='path to concept vocab'
        )
        group.add_argument(
            '--relations', 
            type=str, 
            default='/users/FrankVerhoef/Programming/Project_AI/multigen/data/relation.txt', 
            help='path to relation names'
        )
        group.add_argument(
            '--dataset-concepts', 
            type=str, 
            default='/users/FrankVerhoef/Programming/Project_AI/multigen/data/anlg/total_concepts.txt', 
            help='path to dataset vocab'
        )
        group.add_argument(
            '--kg', 
            type=str, 
            default='/users/FrankVerhoef/Programming/Project_AI/multigen/data/cpnet_25.graph', 
            help='path to knowledge graph'
        )
        group.add_argument(
            '-hid', '--hidden-size', type=int, default=256, help='Hidden size.'
        )
        group.add_argument(
            '-emb', '--embedding-size', type=int, default=768, help='Hidden size.'
        )
        group.add_argument(
            "--source_length",
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
            "--max-memory-size",
            type=int,
            default=400,
            help="Maximum number of related concepts to include."
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
            "--aggregate-method",
            type=str,
            default="max",
            choices=["avg", "max"],
            help="How to aggregate probabilities on graph nodes."
        )
        return parser


    def __init__(self, opt):
        super().__init__(opt)

        self.model_vocab = self.dict.keys()
        self.max_memory_size = self.opt['max_memory_size']
        self.kg = ConceptGraph(self.opt['concepts'], self.opt['relations'], self.opt['dataset_concepts'], self.opt['kg'])
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


    def train_step(self, batch):
        pass

    # def eval_step(self, batch):
    #     for i, ex in enumerate(batch.concept_labels):
    #         print("Eval_Step: ", i, ex)
    #     # for each row in batch, convert tensor to back to text strings
    #     return Output([self.dict.vec2txt(row) for row in batch.text_vec])


    def observe(self, observation):
        logging.debug('=== Observation ===\n{}'.format(observation['text']))

        # Match with concepts in knowledge graph
        concepts = self.kg.match_mentioned_concepts(observation['text'], ' '.join(observation['eval_labels']))
        # for k, v in concepts.items():
        #     print(k, v)
        related_concepts = self.kg.find_neighbours_frequency(concepts['qc'], concepts['ac'], T=2, max_B=100)[0]
        # print("Related concepts: ")
        # for k, v in related_concepts.items():
        #     print(k, v[:10])
        
        filtered_data = filter_directed_triple(related_concepts, max_concepts=400, max_triples=1000)

        # TODO: Think of alternative way to match concepts to tokens
        observation['related_concepts'] = filtered_data['concepts']
        observation['concept_ids'] = torch.LongTensor([self.dict.txt2vec(c)[0] for c in filtered_data['concepts']])
        observation['concept_labels'] = torch.LongTensor(filtered_data['labels'])
        observation['distances'] = torch.LongTensor(filtered_data['distances'])
        observation['relations'] = torch.LongTensor(filtered_data['relations'])
        observation['head_ids'] = torch.LongTensor(filtered_data['head_ids'])
        observation['tail_ids'] = torch.LongTensor(filtered_data['tail_ids'])
        observation['triple_labels'] = torch.LongTensor(filtered_data['triple_labels'])

        super().observe(observation)

        logging.debug("Found {} related concepts and {} relations".format(
            len(observation['related_concepts']), len(observation['relations']))
        )
        logging.debug("\t{}".format(', '.join(observation['related_concepts'][:10])))

        return observation


    def act(self):

        logging.debug("=== Action === ")
        reply = super().act()

        return reply

    def batchify(self, obs_batch, sort=False):
        batch = super().batchify(obs_batch, sort=sort)

        batch['concept_ids'], _ = self._pad_tensor([obs_batch[i]['concept_ids'] for i in batch.valid_indices])
        batch['concept_labels'], _ = self._pad_tensor([obs_batch[i]['concept_labels'] for i in batch.valid_indices])
        batch['relations'], _ = self._pad_tensor([obs_batch[i]['relations'] for i in batch.valid_indices])
        batch['head_ids'], _ = self._pad_tensor([obs_batch[i]['head_ids'] for i in batch.valid_indices])
        batch['tail_ids'], _ = self._pad_tensor([obs_batch[i]['tail_ids'] for i in batch.valid_indices])
        batch['vocab_map'] = torch.LongTensor(self.vocab_map)
        batch['map_mask'] = torch.LongTensor(self.map_mask)

        return batch


    def _encoder_input(self, batch):
        return self._model_input(batch)


    def _model_input(self, batch):
        return (
            batch.text_vec,
            batch.concept_ids,
            batch.relations,
            batch.head_ids,
            batch.tail_ids,
            batch.vocab_map,
            batch.map_mask
        )


    def _get_initial_decoder_input(self, bsz, beam_size, dev):
        return (
            torch.LongTensor([self.dict.start_idx])
            .expand(bsz * beam_size, 1)
            .to(dev)
        )


    def compute_loss(self, batch, return_output):

        """
        Compute and return the loss for the given batch.

        Easily overridable for customized loss functions.

        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')

        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        scores, preds, *_ = model_output
        score_view = scores.reshape(-1, scores.size(-1))
        loss = self.criterion(score_view, batch.label_vec.view(-1))

         # actually do backwards loss
        loss = loss.sum()
        # loss /= target_tokens.sum()  # average loss per token

        if return_output:
            return (loss, model_output)
        else:
            return loss


    def build_criterion(self):
        """
        Construct and return the loss function.
        """
        return self.KG_loss


    def KG_loss(self, hybrid_probs, labels):

        gen_loss_fn = nn.NLLLoss(ignore_index=-1)
        hybrid_probs_clamp = hybrid_probs.clamp(min=1e-5)
        gen_loss = gen_loss_fn(hybrid_probs_clamp.log().view(-1, hybrid_probs.size(-1)), labels.view(-1))
        assert(not torch.isinf(gen_loss).any().item())

        loss = gen_loss
        
        return loss

#    def KG_loss(self, hybrid_probs, labels, gate, gate_mask, gate_label, triple_score, triple_label):

        # gate_loss_fn = nn.BCELoss(weight=gate_mask.view(-1), reduction='mean')
        # gate_loss = gate_loss_fn(gate.view(-1), gate_label.view(-1).float())

        # triple_mask = (triple_label != -1).unsqueeze(1).expand_as(triple_score).float()
        # triple_label = triple_label.unsqueeze(1).expand_as(triple_score) * triple_mask
        # triple_loss_fn = nn.BCELoss(weight=triple_mask.view(-1), reduction='mean')
        # triple_loss = triple_loss_fn(triple_score.view(-1), triple_label.view(-1).float())

        # gen_loss_fn = nn.NLLLoss(ignore_index=-1, reduction='mean')
        # hybrid_probs_clamp = hybrid_probs.clamp(min=1e-5)
        # gen_loss = gen_loss_fn(hybrid_probs_clamp.log().view(-1, hybrid_probs.size(-1)), labels.view(-1))
        # assert(not torch.isinf(gen_loss).any().item())

        # loss = gen_loss + self.alpha * gate_loss + self.beta * triple_loss
        
        # return loss, gen_loss, gate_loss, triple_loss