from parlai.agents.hugging_face.gpt2 import Gpt2Agent

import torch
import torch.nn as nn
import torch.nn.functional as F
from parlai.utils.misc import warn_once
from parlai.utils.torch import padded_tensor, neginf
from parlai.utils.strings import colorize
from parlai.core.metrics import AverageMetric
from parlai.core.torch_agent import Output
from parlai.core.torch_generator_agent import PPLMetric, GreedySearch, _PathSelection, _HypothesisTail

from parlai.agents.knowledge_grounded_generator.kg_utils import NOCONCEPT_TOKEN, NORELATION_TOKEN, ConceptGraph, blacklist
from parlai.agents.knowledge_grounded_generator.kg_model import KnowledgeGroundedModel
import parlai.utils.logging as logging


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
        self.show_token_details = self.show_token_details \
            or 'concepts' in opt.get('display_add_fields', '') \
            or 'probs' in opt.get('display_add_fields', '') 
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
        logging.debug('=== KGG AGENT - OBSERVE ===')

        observation = super().observe(observation)

        if 'full_text' in observation.keys():
            text = observation['full_text']
        elif 'text' in observation.keys():
            text = observation['text']
        else:
            text = ''
        logging.debug('Text:{}'.format(text))

        if 'labels' in observation.keys():
            labels = observation['labels']
        elif 'eval_labels' in observation.keys():
            labels = observation['eval_labels']
        else:
            labels = []
        logging.debug("Labels: {}".format(labels))

        # Match input text and label with concepts in knowledge graph
        concepts = self.kg.match_mentioned_concepts(text, ' '.join(labels), self.overlapping_concepts)
        logging.debug("Concepts: {} + {}".format(concepts['source_concepts'], concepts['target_concepts']))

        # Find related concepts and connecting triples
        related_concepts = self.kg.find_neighbours_nx(concepts['source_concepts'], concepts['target_concepts'], num_hops=self.num_hops, max_B=100)
        filtered_data = self.kg.filter_directed_triple(related_concepts, max_concepts=self.max_concepts, max_triples=self.max_triples)

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

        # Construct list with gate_labels
        target_concept_ids = [self.dict.txt2vec(' ' + c)[0] for c in concepts['target_concepts']]
        label_ids = self.dict.txt2vec(labels[0]) if len(labels) > 0 else []
        gate_labels = [1 if x in target_concept_ids else 0 for x in label_ids] + [0] # add 0 for end-token

        # Info about the related concepts
        concept_token_ids = [self.dict.txt2vec(' ' + self.kg.id2concept[id])[0] for id in filtered_data['concept_ids']]
        observation['concept_token_ids'] = torch.LongTensor(concept_token_ids)
        observation['concept_labels'] = torch.LongTensor(filtered_data['labels'])
        observation['distances'] = torch.LongTensor(filtered_data['distances'])
        observation['gate_labels'] = torch.LongTensor(gate_labels)

        # Info how to map concepts to vocab
        observation['vocab_map'], observation['map_mask'] = self.build_vocab_map(concept_token_ids)

        # Info about relations to related concepts
        observation['relation_ids'] = torch.LongTensor(filtered_data['relation_ids'])
        observation['head_idx'] = torch.LongTensor(filtered_data['head_idx'])
        observation['tail_idx'] = torch.LongTensor(filtered_data['tail_idx'])
        observation['triple_labels'] = torch.LongTensor(filtered_data['triple_labels'])

        # Add metrics about observation
        num_triples = len(related_concepts['triples'])
        self.global_metrics.add(
            'triple_trunc', 
            AverageMetric((num_triples - len(filtered_data['head_idx'])) / max(num_triples,1), 1)
        )
        self.global_metrics.add(
            'gate_label', 
            AverageMetric(sum(gate_labels)/max(len(label_ids), 1), 1)
        )
        self.global_metrics.add(
            'triple_label', 
            AverageMetric(sum(observation['triple_labels'])/max(len(observation['triple_labels']), 1), 1)
        )

        return observation


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


    def compute_loss(self, batch, return_output=False):
        """
        Compute and return the loss for the given batch.
        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """

        def show_teacher_forcing(text_vec, preds, label_vec, vec2txt_fn):
            """
            Helper function to check teacher forcing
            """
            s = ""
            for i, (t, p, c, l) in enumerate(zip(text_vec, preds, is_concept, label_vec)):
                for num in range(len(l)):
                    s += "Generate {}/{}: ".format(i, num)
                    s += colorize(vec2txt_fn(t), 'text')
                    s += colorize(vec2txt_fn(l[:num]), 'labels')
                    if p[num] == l[num]:
                        s += colorize(vec2txt_fn([p[num]]), 'bold_text')
                    else:
                        s += colorize(vec2txt_fn([p[num]]), 'highlight')
                    s += "\n"
            return s    

        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')

        logging.debug("=== KGG AGENT - COMPUTE LOSS === ")
        logging.debug("Size input {}, label {}".format(batch.text_vec.shape, batch.label_vec.shape))
    
        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        scores, preds, encoder_states, triple_prob, gate, is_concept, lm_probs, concept_probs = model_output

        # logging.debug("Teacher forcing\n{}".format(show_teacher_forcing(batch.text_vec, preds, batch.label_vec, self.dict.vec2txt)))

        combined_loss, gen_loss, triple_loss, gate_loss = self.criterion(
            scores, batch.label_vec, 
            triple_prob, batch.triple_labels, 
            gate, batch.gate_labels
        )

        # Record metrics
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


    def eval_step(self, batch):
        """
        Evaluate a single batch of examples.
        """
        logging.debug('=== KGG AGENT - EVAL_STEP ===')
        if batch.text_vec is None and batch.image is None:
            return
        if batch.text_vec is not None:
            bsz = batch.text_vec.size(0)
        else:
            bsz = len(batch.image)
        self.model.eval()
        cand_scores = None
        token_losses = None
        text_token_info = None
        concept_info = None
        probs_info = None

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            loss, model_output = self.compute_loss(batch, return_output=True)
            if self.show_token_details:
                token_losses = self._construct_label_token_losses(batch.label_vec, model_output)

        beam_preds_scores = None
        preds = None
        if self.skip_generation:
            warn_once("--skip-generation true produces limited metrics")
        else:
            maxlen = self.label_truncate or 64
            prefix_tokens = self.get_prefix_tokens(batch)
            beam_preds_scores, beams = self._generate(
                batch, 
                self.beam_size, 
                maxlen, 
                prefix_tokens=prefix_tokens
            )
            preds, _, _ = zip(*beam_preds_scores)
            self._add_generation_metrics(batch, preds)

            # bsz x beamsize
            beam_texts = []
            beam_texts_token_info = []
            for beam in beams:
                beam_texts.append([])
                if self.show_token_details:
                    beam_texts_token_info.append([])

                for tokens, score, token_metadata in beam.get_rescored_finished():
                    try:
                        if self.show_token_details:
                            beam_texts_token_info[-1].append(
                                self._construct_generated_token_details(tokens, token_metadata)
                            )
                        beam_texts[-1].append((self._v2t(tokens), score.item()))
                    except KeyError:
                        logging.error("Decoding error: %s", tokens)
                        continue

        cand_choices = None
        cand_scores = None
        if self.rank_candidates:
            cand_choices, cand_scores = self.rank_eval_label_candidates(batch, bsz)

        text = (
            [self._v2t(pred_data[0]) for pred_data in beam_preds_scores]
            if beam_preds_scores is not None
            else None
        )

        if self.show_token_details and beam_preds_scores is not None:
            text_token_info = []
            concept_info = []
            probs_info = []
            for beam_text_token_info in beam_texts_token_info:
                text_token_info.append(beam_text_token_info[0])
                concept_info.append([
                    (token_info[0], token_info[1].get('gate', 0.0), token_info[1].get('is_concept', 0))
                    for token_info in text_token_info[-1]
                ])
                probs_info.append([
                    (
                        token_info[0], 
                        [(self._v2t([token]), prob) for token, prob in token_info[1].get('lm_probs')], 
                        [(self._v2t([token]), prob) for token, prob in token_info[1].get('concept_probs')],
                        token_info[1].get('gate', 0.0)
                    )
                    for token_info in text_token_info[-1]
                ])

        if text and self.compute_tokenized_bleu:
            # compute additional bleu scores
            self._compute_fairseq_bleu(batch, preds)

        retval = Output(text, cand_choices, token_losses=token_losses, cand_scores=cand_scores)
        if not self.skip_generation:
            retval.beam_texts = beam_texts
            retval.beam_texts_token_info = beam_texts_token_info
            retval.text_token_info = text_token_info
            retval.concepts = concept_info
            retval.probs = probs_info
            
        return retval


    def _generate(self, batch, beam_size, max_ts, prefix_tokens = None):
        """
        Generate an output with beam search.

        Depending on the options, this may perform greedy/topk/nucleus generation.

        :param Batch batch:
            Batch structure with input and labels
        :param int beam_size:
            Size of each beam during the search
        :param int max_ts:
            the maximum length of the decoded sequence
        :param prefix_tokens:
            if given, a tensor of tokens that must begin the decoded sequence.

        :return:
            tuple (beam_pred_scores, beams)
            - beam_preds_scores: list of (prediction, score, token_metadata) tuples for each sample in Batch
            - beams :list of Beam instances defined in Beam class, can be used for any
              following postprocessing, e.g. dot logging.
        """
        logging.debug("=== KGG AGENT - _GENERATE ===")
        logging.debug("Beamsize = {}, Maxlen = {}".format(beam_size, max_ts))
        model = self.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        encoder_states = model.encoder(*self._encoder_input(batch))
        if batch.text_vec is not None:
            dev = batch.text_vec.device
        else:
            assert batch.label_vec is not None, "need label_vec for _generate"
            dev = batch.label_vec.device

        bsz = batch.batchsize
        if batch.text_vec is not None:
            batchsize = batch.batchsize
            batch_context_list = self._get_batch_context(batch).tolist()
            beams = [
                self._treesearch_factory(dev, verbose=self.show_token_details)
                .set_batch_context(batch_context_list, batch_idx)
                .set_block_list(self.beam_block_list)
                for batch_idx in range(batchsize)
            ]
        else:
            beams = [
                self._treesearch_factory(dev, verbose=self.show_token_details)
                for _ in range(bsz)
            ]

        # repeat encoder outputs and decoder inputs
        decoder_input = self._get_initial_decoder_input(bsz, beam_size, dev)
        inds = torch.arange(bsz).to(dev).unsqueeze(1).repeat(1, beam_size).view(-1)
        encoder_states = model.reorder_encoder_states(encoder_states, inds)
        incr_state = None

        for _ts in range(max_ts):
            if all((b.is_done() for b in beams)):
                # exit early if possible
                break

            score, incr_state = model.decoder(decoder_input, encoder_states, incr_state)
            input_ids, gpt_states, triple_prob, gate, is_concept, lm_probs, concept_probs, kg_mem = incr_state

            # only need the final hidden state to make the word prediction
            score = score[:, -1:, :]
            score = model.output(score)

            # score contains softmax scores for bsz * beam_size samples
            score = score.view(bsz, beam_size, -1)
            if self.temperature != 1.0:
                score.div_(self.temperature)

            # force to fp32 to avoid overflow issues during search calculations
            score = F.log_softmax(score, dim=-1, dtype=torch.float32)  # type: ignore
            if prefix_tokens is not None and _ts < prefix_tokens.size(1):
                # generate prefix_tokens for every timestep that they exist
                # achieve by setting score of all other tokens to be -inf
                prefix_toks = prefix_tokens[:, _ts]
                prefix_mask = torch.ones_like(score, dtype=torch.bool)
                prefix_mask[:, :, prefix_toks] = False  # everything except prefix toks should be neginf
                score[prefix_mask] = neginf(score.dtype)

            for i, b in enumerate(beams):
                if not b.is_done():
                    b.advance(score[i], gate[i][-1], is_concept[i][-1], lm_probs[i][-1], concept_probs[i][-1])
            incr_state_inds = torch.cat([
                beam_size * i + b.get_backtrack_from_current_step()
                for i, b in enumerate(beams)
            ])
            incr_state = model.reorder_decoder_incremental_state(incr_state, incr_state_inds)
            selection = torch.cat([b.get_output_from_current_step() for b in beams]).unsqueeze(-1)
            decoder_input = self._get_next_decoder_input(decoder_input, selection, incr_state_inds)

        # get all finalized candidates for each sample (and validate them)
        n_best_beam_preds_scores = [b.get_rescored_finished() for b in beams]

        if hasattr(self, '_rerank_beams'):
            n_best_beam_preds_scores = self._rerank_beams(batch, n_best_beam_preds_scores)

        # get the top prediction for each beam (i.e. minibatch sample)
        beam_preds_scores = [n_best_list[0] for n_best_list in n_best_beam_preds_scores]

        return beam_preds_scores, beams


    def _treesearch_factory(self, device, verbose=False):
        method = self.opt.get('inference', 'greedy')
        beam_size = self.opt.get('beam_size', 1)
        if method == 'greedy':
            return KGG_GreedySearch(
                beam_size,
                min_length=0,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=device,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Can't use inference method {method}")


class KGG_GreedySearch(GreedySearch):
    """
    Greedy search.

    Picks the highest probability utterance at each step.  Only works with
    --beam-size 1.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.verbose:
            for i in range(self.beam_size):
                self.token_details[i][0].update({
                    "gate": 0.0,
                    "is_concept": 0,
                    "lm_probs": [(self.bos,1.0)] + 4 *[(self.pad, 0.0)],
                    "concept_probs": 5 * [(self.pad, 0.0)]
                })


    def advance(self, logprobs, gate, is_concept, lm_probs, concept_probs):
        """
        Advance the beam one step.
        """
        current_length = len(self.all_scores) - 1
        if current_length < self.min_length:
            # penalize all eos probs to make it decode longer
            for hyp_id in range(logprobs.size(0)):
                logprobs[hyp_id][self.eos] = neginf(logprobs.dtype)

        if self.scores is None:
            self.scores = torch.zeros(1).type_as(logprobs).to(logprobs.device)

        # penalize hypotheses ending in EOS on the prior scores (self.scores) level
        # this is related to search which uses prior scores (self.scores) (e.g. beam)
        for hyp_id, token in enumerate(self.outputs[-1]):
            if token == self.eos:
                self.scores[hyp_id] = neginf(self.scores.dtype)

        # beam blocking
        if self.block_ngram > 0:
            logprobs = self._block_ngrams(self.block_ngram, logprobs, None)

        logprobs = self._block_block_list(logprobs)

        if self.context_block_ngram > 0:
            if self.context is None:
                raise ValueError("Must use TreeSearch.set_context to use context blocking.")
            logprobs = self._block_ngrams(self.context_block_ngram, logprobs, self.context)

        path_selection = self.select_paths(logprobs, self.scores, current_length, gate, is_concept, lm_probs, concept_probs)
        self.scores = path_selection.scores
        # use clone() here to ensure that self.all_scores will not be changed
        # later due to any penalties to self.scores
        self.all_scores.append(self.scores.clone())

        self.outputs.append(path_selection.token_ids)
        self.bookkeep.append(path_selection.hypothesis_ids)
        tok_id_list = path_selection.token_ids.tolist()
        self.partial_hyps = [
            self.partial_hyps[path_selection.hypothesis_ids[i]] + [tok_id_list[i]]
            for i in range(self.beam_size)
        ]

        if self.verbose:
            assert path_selection.token_details
            assert self.token_details
            for i in range(self.beam_size):
                self.token_details[i].append(path_selection.token_details[i])

        #  check new hypos for eos label, if we have some, add to finished
        for hypid in range(self.beam_size):
            if self.outputs[-1][hypid] == self.eos:
                if self.scores[hypid] <= neginf(self.scores.dtype):
                    continue
                #  this is finished hypo, adding to finished

                eostail = _HypothesisTail(
                    timestep=len(self.outputs) - 1,
                    hypid=hypid,
                    score=self.all_scores[-1][hypid],
                    tokenid=self.eos,
                    token_details=self.token_details[hypid][-1]
                    if self.token_details is not None
                    else None,
                )
                self.finished.append(eostail)
                self.n_best_counter += 1

        if self.outputs[-1][0] == self.eos:
            self.eos_top = True
            if self.eos_top_ts is None:
                self.eos_top_ts = len(self.outputs) - 1


    def select_paths(self, logprobs, prior_scores, current_length, gate, is_concept, lm_probs, concept_probs) -> _PathSelection:
        path_selection = super().select_paths(logprobs, prior_scores, current_length)

        if self.verbose:
            lm_topk = torch.topk(lm_probs, 5)
            concept_topk = torch.topk(concept_probs, 5)
            concept_token_details = {
                "is_concept": is_concept.item(),
                "lm_probs": [(lm_topk.indices[i].item(), lm_topk.values[i].item()) for i in range(5)],
                "concept_probs": [(concept_topk.indices[i].item(), concept_topk.values[i].item()) for i in range(5)],
                "gate": gate.item()
            }
            path_selection.token_details[0].update(concept_token_details)

        return path_selection
