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
        group.add_argument(
            "--fixed-lm",
            type=bool,
            default=False,
            help="Freeze the weights of the GPT2 language model during training."
        )
        group.add_argument(
            "--overlapping-concepts",
            type=str,
            choices=["excl-tgt-in-src", "excl-src-in-tgt", "keep-src-and-tgt"],
            default="excl-src-in-tgt",
            help="How to ensure disjoint sets of concepts."
        )
        group.add_argument(
            "--block-src",
            type=bool,
            default=True,
            help="Blocking source concepts in reasoning stimulates generation of new related concepts."
        )
        return parser


    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        self.num_hops = opt['num_hops']
        self.max_concepts = opt['max_concepts']
        self.max_triples = opt['max_triples']
        self.overlapping_concepts = opt['overlapping_concepts']
        if shared == None:
            self.model_vocab = self.dict.keys()
            self._cache_sorted_dict_ind = sorted(self.dict.ind2tok.keys())
            self.kg = ConceptGraph(opt['kg_datadir'], opt['concepts'], opt['relations'], opt['kg'])
            self.matched_concepts = self.match_dataset_concepts(opt['kg_datadir'] + opt['dataset_concepts'])
            self.kg.build_reduced_graph(self.matched_concepts)
            logging.info("Initialized KnowledgeGroundedGeneratorAgent")
        else:
            self.model_vocab = shared['model_vocab']
            self._cache_sorted_dict_ind = shared['cache_dict_ind'] 
            self.kg = shared['kg']
            self.matched_concepts = shared['matched_concepts']
            logging.info("Initialized KnowledgeGroundedGeneratorAgent [shared]")


    def share(self):
        """
        Share fields from parent as well as useful objects in this class.
        """
        shared = super().share()
        shared['model_vocab'] = self.model_vocab
        shared['cache_dict_ind'] = self._cache_sorted_dict_ind
        shared['kg'] = self.kg
        shared['matched_concepts'] = self.matched_concepts

        return shared


    def build_model(self):

        model = KnowledgeGroundedModel(self.opt, self.dict)
        return model


    def match_dataset_concepts(self, concepts_path):
        """
        Returns the set of concepts that appear in both the dataset and in the knowledge graph, 
        and are not on the blacklist
        """

        # total_concepts is the set of concepts that appears in the dataset (train and validation dialogues)
        with open(concepts_path, 'r') as f:
            total_concepts = set([line[:-1] for line in f])

        # graph_concepts is the list of concepts in the knowledge graph
        graph_concepts = [self.kg.id2concept[id] for id in self.kg.graph.nodes]

        # matched concepts is the dataset concepts, minus words on the blacklist (e.g. auxiliary verbs) 
        # that also appear in the knowledge graph
        matched_concepts = (total_concepts - blacklist).intersection(graph_concepts)

        logging.info("Matched {} of {} dataset tokens with {} concepts in knowledgegraph".format(
            len(matched_concepts), len(total_concepts), len(graph_concepts)
        ))
        return matched_concepts


    def build_vocab_map(self, concept_token_ids):
        """
        The vocab map and associated mask are a mapping between the GPT2 vocabulary and the KG concepts
        in the current observation.
        At each position in the vocab map, the value is the index in the list of concepts that are 
        present in the observation. The vocab mask and map mask are used in the KGG-model to map the 
        calculated concept-scores back to token-scores in the GPT2 vocabulary.
        """
        vocab_map = torch.zeros(len(self._cache_sorted_dict_ind), dtype=torch.long)
        map_mask = torch.zeros_like(vocab_map)
        for i, token_id in enumerate(self._cache_sorted_dict_ind):
            try: 
                pos = concept_token_ids.index(token_id)
                vocab_map[i] = pos
                map_mask[i] = 1
            except ValueError:
                pass

        return vocab_map, map_mask


    def observe(self, observation):
        logging.debug('=== Observation ===')

        start = timer.time()
        observation = super().observe(observation)
        time_super = timer.time()

        if 'full_text' in observation.keys():
            text = observation['full_text']
        elif 'text' in observation.keys():
            text = observation['text']
        else:
            text = ''
        logging.debug('Text:{}'.format(text))

        if 'labels' in observation.keys():
            labels = observation['labels']
            labels_vec = observation['labels_vec']
        elif 'eval_labels' in observation.keys():
            labels = observation['eval_labels']
            labels_vec = observation['eval_labels_vec']
        else:
            labels = []
            labels_vec = torch.tensor([])
        logging.debug("Labels: {}".format(labels))

        # Match with concepts in knowledge graph
        concepts = self.kg.match_mentioned_concepts(text, ' '.join(labels), self.overlapping_concepts)
        logging.debug("Concepts: {} + {}".format(concepts['qc'], concepts['ac']))
        related_concepts = self.kg.find_neighbours_nx(concepts['qc'], concepts['ac'], num_hops=self.num_hops, max_B=100)
        time_related = timer.time()
        filtered_data = self.kg.filter_directed_triple(related_concepts, max_concepts=self.max_concepts, max_triples=self.max_triples)
        time_filter = timer.time()

        # Construct list with gate_labels
        target_concept_ids = [self.dict.txt2vec(' ' + c)[0] for c in concepts['ac']]
        label_ids = self.dict.txt2vec(labels[0]) if len(labels) > 0 else []
        gate_labels = [1 if x in target_concept_ids else 0 for x in label_ids] + [0] # add 0 for end-token

        # Info about the related concepts
        concept_token_ids = [self.dict.txt2vec(' ' + self.kg.id2concept[id])[0] for id in filtered_data['concept_ids']]
        time_t2v = timer.time()
        observation['concept_token_ids'] = torch.LongTensor(concept_token_ids)
        observation['concept_labels'] = torch.LongTensor(filtered_data['labels'])
        observation['distances'] = torch.LongTensor(filtered_data['distances'])
        observation['gate_labels'] = torch.LongTensor(gate_labels)

        # Info how to map concepts to vocab
        observation['vocab_map'], observation['map_mask'] = self.build_vocab_map(concept_token_ids)
        time_map = timer.time()

        # Info about relations to related concepts
        observation['relation_ids'] = torch.LongTensor(filtered_data['relation_ids'])
        observation['head_idx'] = torch.LongTensor(filtered_data['head_idx'])
        observation['tail_idx'] = torch.LongTensor(filtered_data['tail_idx'])
        observation['triple_labels'] = torch.LongTensor(filtered_data['triple_labels'])

        # logging.debug("Related concepts {}: {}".format(
        #     len(filtered_data['concept_ids']), 
        #     self.kg.formatted_concepts_string(filtered_data, 10)
        # ))
        # logging.debug("Translated concepts: {}".format([
        #     (self.kg.id2concept[id], self.dict.vec2txt([self.dict.txt2vec(' ' + self.kg.id2concept[id])[0]]))
        #     for id in filtered_data['concept_ids']
        # ]))
        # logging.debug("Relations {}: {}".format(
        #     len(observation['head_idx']),
        #     self.kg.formatted_triples_string(filtered_data, 5)
        # ))
        time_obs = timer.time()
        # logging.debug("Times super {:6.4f}, related {:6.4f}, filter {:6.4f}, t2v {:6.4f}, map {:6.4f}, obs {:6.4f}".format(
        #     time_super-start, time_related-time_super, time_filter-time_related, time_t2v-time_filter,
        #     time_map-time_t2v, time_obs-time_map))
        logging.debug("Observation time: {:6.4f}".format(timer.time() - start))

        ## log statistics
        # Number of concepts in the input
        # self.global_metrics.add('src_cpt', AverageMetric(len(concepts['qc']), 1))
        # Fraction of target tokens that is a concept reachable within max number of hops
        logging.debug("Target: {}".format(''.join([
            '\033[48;5;{}m'.format(46 if gate_labels[i] == 1 else 231) \
            + self.dict.vec2txt([token_id]) \
            + '\033[0;0m'
            for i, token_id in enumerate(labels_vec)
        ])))
        self.global_metrics.add(
            'gate_label', 
            AverageMetric(sum(gate_labels)/max(len(label_ids), 1), 1)
        )
        self.global_metrics.add(
            'triple_label', 
            AverageMetric(sum(observation['triple_labels'])/max(len(observation['triple_labels']), 1), 1)
        )

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
        logging.debug("Size input {}, label {}".format(batch.text_vec.shape, batch.label_vec.shape))
        start = timer.time()
    
        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        time_fwd = timer.time()
        scores, preds, encoder_states, triple_prob, gate, is_concept, lm_probs, concept_probs = model_output
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
        time_loss = timer.time()
        logging.debug("\tforward {}".format(time_fwd-start))
        logging.debug("\tloss_fn {}".format(time_loss-time_fwd))

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
        scores, preds, encoder_states, triple_prob, gate, is_concept, lm_probs, concept_probs = model_output
        score_view = scores.clamp(min=1e-5).reshape(-1, scores.size(-1))
        losses = self.criterion.gen_loss_fn(score_view.log(), labels.view(-1)).view(len(labels), -1)

        # Zip decoded tokens with losses
        token_losses = []
        for i, l in enumerate(labels):
            token_losses.append(
                list(
                    zip(
                        [self.dict.vec2txt(token) for token in l.tolist()],
                        losses[i].tolist(),
                    )
                )
            )
        return token_losses


    def _construct_generated_token_details(self, tokens, tokens_metadata):
        tokens_as_txt = [self.dict.vec2txt(token) for token in tokens.tolist()]
        return list(zip(tokens_as_txt, tokens_metadata))


    def build_criterion(self):
        """
        Construct and return the loss function.
        Uses parameters alpha and beta to determine how much of gate loss (x alpha) and triple loss (x beta)
        is added to the final loss
        """
        return KG_loss(ignore_index=self.NULL_IDX, invalid=-1, alpha = self.opt['alpha'], beta = self.opt['beta'])
