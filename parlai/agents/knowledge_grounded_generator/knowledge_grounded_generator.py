from parlai.agents.hugging_face.gpt2 import Gpt2Agent

import torch
import torch.nn as nn
import torch.nn.functional as F
from parlai.utils.torch import padded_tensor
from parlai.core.metrics import AverageMetric
from parlai.core.torch_generator_agent import PPLMetric

from parlai.agents.knowledge_grounded_generator.kg_utils import NOCONCEPT_TOKEN, NORELATION_TOKEN, ConceptGraph, blacklist
from parlai.agents.knowledge_grounded_generator.kg_model import KnowledgeGroundedModel
import parlai.utils.logging as logging
import time as timer
from parlai.utils.strings import colorize

class KG_loss(nn.Module):

    def __init__(self, ignore_index, invalid, alpha, beta):
        super().__init__()
        self.ignore_index = ignore_index
        self.invalid = invalid
        self.alpha = alpha
        self.beta = beta
        self.gen_loss_fn = nn.NLLLoss(ignore_index=self.ignore_index, reduction='none')

    def forward(self, lm_probs, labels, triple_prob, triple_labels, gate, gate_labels):
        B = lm_probs.size(0)

        # Compute generation loss
        num_target_tokens = labels.ne(self.ignore_index).long().sum(dim=-1)
        probs_clamp = lm_probs.clamp(min=1e-5)
        gen_loss_token = self.gen_loss_fn(probs_clamp.log().view(-1, lm_probs.size(-1)), labels.view(-1)).view(B, -1)
        gen_loss = gen_loss_token.sum(dim=-1) / num_target_tokens.clamp(min=1)

        # Compute triple loss
        triple_mask = (triple_labels != self.invalid).unsqueeze(1).expand_as(triple_prob).float()
        num_valid_triples = triple_mask.sum(dim=(-2,-1))
        triple_labels = triple_labels.unsqueeze(1).expand_as(triple_prob) * triple_mask
        triple_loss_fn = nn.BCELoss(weight=triple_mask, reduction='none')
        triple_loss_triple = triple_loss_fn(triple_prob, triple_labels.float()).view(B, -1)
        triple_loss = triple_loss_triple.sum(dim=-1) / num_valid_triples.clamp(min=1)

        # Compute gate loss   
        gate_mask = (gate_labels != self.invalid).float()
        gate_labels.masked_fill_((gate_labels == self.invalid), 0)
        lm_mask = (gate_labels.sum(1) != 0).float().unsqueeze(1)
        gate_mask = lm_mask.expand_as(gate_labels) * gate_mask
        num_valid_gates = gate_mask.sum(dim=-1)
        gate_loss_fn = nn.BCELoss(weight=gate_mask, reduction='none')
        gate_loss_token = gate_loss_fn(gate.view(B, -1), gate_labels.float()).view(B, -1)
        gate_loss = gate_loss_token.sum(dim=-1) / num_valid_gates.clamp(min=1)

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
            "--num-hops",
            type=int,
            default=2,
            help="Number of hops in the graph to look for related concepts."
        )
        group.add_argument(
            "--max-concepts",
            type=int,
            default=256,
            help="Maximum number of related concepts to include."
        )
        group.add_argument(
            "--max-triples",
            type=int,
            default=768,
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

        self.num_hops = opt['num_hops']
        self.max_concepts = opt['max_concepts']
        self.max_triples = opt['max_triples']
        if shared == None:
            self.model_vocab = self.dict.keys()
            self.kg = ConceptGraph(opt['kg_datadir'], opt['concepts'], opt['relations'], opt['kg'])
            self.matched_concepts = self.match_dataset_concepts(opt['kg_datadir'] + opt['dataset_concepts'])
            self.kg.build_reduced_graph(self.matched_concepts)
            logging.info("Initialized KnowledgeGroundedGeneratorAgent")
        else:
            self.model_vocab = shared['model_vocab']
            self.kg = shared['kg']
            self.matched_concepts = shared['matched_concepts']
            logging.info("Initialized KnowledgeGroundedGeneratorAgent [shared]")


    def share(self):
        """
        Share fields from parent as well as useful objects in this class.
        """
        shared = super().share()
        shared['model_vocab'] = self.model_vocab
        shared['kg'] = self.kg
        shared['matched_concepts'] = self.matched_concepts

        return shared


    def build_model(self):

        model = KnowledgeGroundedModel(self.opt, self.dict)
        return model


    def match_dataset_concepts(self, concepts_path):

        with open(concepts_path, 'r') as f:
            total_concepts = set([line[:-1] for line in f])
        graph_concepts = [self.kg.id2concept[id] for id in self.kg.graph.nodes]
        matched_concepts = (total_concepts - blacklist).intersection(graph_concepts)
        logging.debug("Matched {} of {} dataset tokens with concepts in knowledgegraph".format(len(matched_concepts), len(total_concepts)))
        return matched_concepts


    def build_vocab_map(self, concept_token_ids):

        vocab_map, map_mask = [], []
        for token_id in sorted(self.dict.ind2tok.keys()):
            try: 
                pos = concept_token_ids.index(token_id)
                vocab_map.append(pos)
                map_mask.append(1)
            except:
                vocab_map.append(0)
                map_mask.append(0)
        assert(len(vocab_map) == len(self.model_vocab))

        return vocab_map, map_mask


    def observe(self, observation):
        logging.debug('=== Observation ===')
        text = observation['text'] if 'text' in observation.keys() else ''
        logging.debug('Text:{}'.format(text))
        start = timer.time()
        observation = super().observe(observation)

        # Match with concepts in knowledge graph
        if 'labels' in observation.keys():
            labels = observation['labels']
        elif 'eval_labels' in observation.keys():
            labels = observation['eval_labels']
        else:
            labels = ''
        logging.debug("Labels: {}".format(labels))
        concepts = self.kg.match_mentioned_concepts(text, ' '.join(labels))
        logging.debug("Concepts: {} + {}".format(concepts['qc'], concepts['ac']))
        related_concepts = self.kg.find_neighbours(concepts['qc'], concepts['ac'], num_hops=self.num_hops, max_B=100)[0]
        filtered_data = self.kg.filter_directed_triple(related_concepts, max_concepts=self.max_concepts, max_triples=self.max_triples)

        # Construct list with gate_labels
        target_concept_ids = [self.dict.txt2vec(' ' + c)[0] for c in concepts['ac']]
        label_ids = self.dict.txt2vec(labels[0])
        gate_labels = [1 if x in target_concept_ids else 0 for x in label_ids] + [0] # add 0 for end-token

        # TODO: Think of alternative way to match concepts to tokens
        # TODO: Truncate or pad to max number of concepts and relations

        # Info about the related concepts
        concept_token_ids = [self.dict.txt2vec(' ' + self.kg.id2concept[id])[0] for id in filtered_data['concept_ids']]
        observation['concept_token_ids'] = torch.LongTensor(concept_token_ids)
        observation['concept_labels'] = torch.LongTensor(filtered_data['labels'])
        observation['distances'] = torch.LongTensor(filtered_data['distances'])
        observation['gate_labels'] = torch.LongTensor(gate_labels)

        # Info how to map concepts to vocab
        vocab_map, map_mask = self.build_vocab_map(concept_token_ids)
        observation['vocab_map'] = torch.LongTensor(vocab_map)
        observation['map_mask'] = torch.LongTensor(map_mask)

        # Info about relations to related concepts
        observation['relation_ids'] = torch.LongTensor(filtered_data['relation_ids'])
        observation['head_idx'] = torch.LongTensor(filtered_data['head_idx'])
        observation['tail_idx'] = torch.LongTensor(filtered_data['tail_idx'])
        observation['triple_labels'] = torch.LongTensor(filtered_data['triple_labels'])

        logging.debug("Related concepts {}: {}".format(
            len(filtered_data['concept_ids']), 
            self.kg.formatted_concepts_string(filtered_data, 10)
        ))
        # logging.debug("Translated concepts: {}".format([
        #     (self.kg.id2concept[id], self.dict.vec2txt([self.dict.txt2vec(' ' + self.kg.id2concept[id])[0]]))
        #     for id in filtered_data['concept_ids']
        # ]))
        logging.debug("Relations {}: {}".format(
            len(observation['head_idx']),
            self.kg.formatted_triples_string(filtered_data, 5)
        ))
        logging.debug("Observation time: {}".format(timer.time() - start))

        ## log statistics
        # Number of concepts in the input
        # self.global_metrics.add('src_cpt', AverageMetric(len(concepts['qc']), 1))
        # Fraction of target tokens that is a concept
        # self.global_metrics.add('gate_label', AverageMetric(sum(gate_labels)/len(label_ids), 1))

        return observation


    def act(self):

        # logging.debug("=== Action === ")
        reply = super().act()

        return reply

    def batchify(self, obs_batch, sort=False):
        batch = super().batchify(obs_batch, sort=sort)

        batch['concept_ids'], _ = padded_tensor(
            [obs_batch[i]['concept_token_ids'] for i in batch.valid_indices],
            pad_idx=self.NULL_IDX          
        )
        batch['concept_labels'], _ = padded_tensor(
            [obs_batch[i]['concept_labels'] for i in batch.valid_indices],
            pad_idx=-1           
        )
        batch['distances'], _ = padded_tensor(
            [obs_batch[i]['distances'] for i in batch.valid_indices],
            pad_idx=0
        )
        batch['relation_ids'], _ = padded_tensor(
            [obs_batch[i]['relation_ids'] for i in batch.valid_indices], 
            pad_idx=self.kg.relation2id[NORELATION_TOKEN]
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
            pad_idx=-1,
            left_padded=True
        )
        batch['vocab_map'] = torch.stack(
            [obs_batch[i]['vocab_map'] for i in batch.valid_indices]
        )
        batch['map_mask'] = torch.stack(
            [obs_batch[i]['map_mask'] for i in batch.valid_indices]
        )

        return batch


    def _encoder_input(self, batch):
        return self._model_input(batch)


    def _model_input(self, batch):
        return (
            batch.text_vec,
            batch.concept_ids,
            batch.concept_labels,
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


    def get_prefix_tokens(self, batch):
        """
        Set prefix tokens to seed decoding at generation time.

        By default, torch_generator_agent does not utilize prefix tokens, but this is
        left overridable by child classes.

        Returned tensor should be of dimension bsz x len(prefix)
        """
        return None # batch.text_vec

    def formatted_response(self, tokens, inserted_concepts, labels):

        s = ""
        for token, is_concept, label in zip(tokens, inserted_concepts, labels):
            if label != self.NULL_IDX:
                if is_concept:
                    s += colorize(self.dict.vec2txt([token]), 'highlight')
                else:
                    s += colorize(self.dict.vec2txt([token]), 'blue')
        return s

    def teacher_forcing(self, text, preds, changed, labels):
        s = colorize(self.dict.vec2txt(text), 'text')
        s += colorize(self.dict.vec2txt(labels[:-1]), 'labels')
        if changed[-1]:
            s += colorize(self.dict.vec2txt([preds[-1]]), 'highlight')
        else:
            s += colorize(self.dict.vec2txt([preds[-1]]), 'highlight2')
        return s        


    def compute_loss(self, batch, return_output=False):
        """
        Compute and return the loss for the given batch.
        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')

        logging.debug("=== KGG Compute Loss === ")
        # logging.debug("\tinput: {}\n\tlabel: {}".format(batch.text_vec, batch.label_vec))
        # logging.debug("Size input {}, label {}".format(batch.text_vec.shape, batch.label_vec.shape))
        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        scores, preds, encoder_states, triple_prob, gate, index_diff = model_output
        # logging.debug("\tpreds: {}".format(preds))
        # for i, (p, c, l) in enumerate(zip(preds, index_diff, batch.label_vec)):
        #     logging.debug("Example {}: {} concepts inserted: {}".format(
        #         i,
        #         int(sum(c)),
        #         self.formatted_response(p, c, l)
        #     ))
        # for i, (t, p, c, l) in enumerate(zip(batch.text_vec, preds, index_diff, batch.label_vec)):
        #     for num in range(1, len(l)+1):
        #         logging.debug("Generate {}/{}: {}".format(
        #             i,
        #             num,
        #             self.teacher_forcing(t, p[:num], c[:num], l[:num])
        #         ))

        combined_loss, gen_loss, triple_loss, gate_loss = self.criterion(
            scores, batch.label_vec, 
            triple_prob, batch.triple_labels, 
            gate, batch.gate_labels
        )

        # record metrics
        notnull = batch.label_vec.ne(self.NULL_IDX)
        num_target_tokens = notnull.long().sum(dim=-1)
        num_correct = ((batch.label_vec == preds) * notnull).sum(dim=-1)
        self.record_local_metric('loss', AverageMetric.many(combined_loss))
        self.record_local_metric('loss_gen', AverageMetric.many(gen_loss))
        self.record_local_metric('loss_triple', AverageMetric.many(triple_loss))
        self.record_local_metric('loss_gate', AverageMetric.many(gate_loss))
        self.record_local_metric('ppl', PPLMetric.many(gen_loss))
        self.record_local_metric('token_acc', AverageMetric.many(num_correct, num_target_tokens))
        loss = combined_loss.sum()

        if return_output:
            return (loss, model_output)
        else:
            return loss


    def _construct_label_token_losses(self, labels, model_output):
        # Get non-aggregated losses
        scores, preds, encoder_states, triple_prob, gate, index_diff = model_output
        score_view = scores.clamp(min=1e-5).reshape(-1, scores.size(-1))
        losses = self.criterion.gen_loss_fn(score_view.log(), labels.view(-1)).view(len(labels), -1)

        # Zip decoded tokens with losses
        token_losses = []
        for i, label in enumerate(labels):
            token_losses.append(
                list(
                    zip(
                        [self.dict[token] for token in label.tolist()],
                        losses[i].tolist(),
                    )
                )
            )
        return token_losses


    def build_criterion(self):
        """
        Construct and return the loss function.
        Uses parameters alpha and beta to determine how much of gate loss (x alpha) and triple loss (x beta)
        is added to the final loss
        """
        return KG_loss(ignore_index=self.NULL_IDX, invalid=-1, alpha = self.opt['alpha'], beta = self.opt['beta'])
