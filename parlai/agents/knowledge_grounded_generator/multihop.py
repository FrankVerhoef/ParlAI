import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_max, scatter_mean, scatter_add
from parlai.agents.hugging_face.gpt2 import GPT2Decoder, HFGPT2Model
from parlai.agents.transformer.modules.encoder import TransformerEncoder
from parlai.core.torch_generator_agent import TorchGeneratorModel
import parlai.utils.logging as logging


class Identity(nn.Module):
    def forward(self, *batch):
        return tuple(batch)
    # def forward(self, text_vec, concept_ids, relations, head_ids, tail_ids, vocab_map, map_mask):
    #     return (text_vec, concept_ids, relations, head_ids, tail_ids, vocab_map, map_mask)


class TripleEncoder(nn.Module):

    def __init__(self, embedding, num_hops, emb_size):
        super().__init__()

        self.num_hops = num_hops
        self.concept_embd = embedding
        self.relation_embd = nn.Embedding(40, emb_size)
        self.W_s = nn.ModuleList([nn.Linear(emb_size, emb_size, bias=False) for _ in range(self.num_hops)]) 
        self.W_n = nn.ModuleList([nn.Linear(emb_size, emb_size, bias=False) for _ in range(self.num_hops)]) 
        self.W_r = nn.ModuleList([nn.Linear(emb_size, emb_size, bias=False) for _ in range(self.num_hops)])

        logging.info("Initialized TripleEncoder")

    def forward(self, concept_ids, relations, head_ids, tail_ids):
        """
        Encodes knowledge triples
        Tensor sizes are:
            B x L1: for concept_ids
            B x L2: for relations, head_ids, tail_ids
            B x L2 x E: for output (triple representation)

            B = batch size
            L1 = number of related concepts (can vary per batch)
            L2 = number of related triples (can vary per batch)
            E = 
        """
        logging.debug("Forward TripleEncoder")
        logging.debug("\tEncoding {} concepts and {} relations".format(concept_ids.size(1), relations.size(1)))

        concept_repr = self.concept_embd(concept_ids)
        rel_repr = self.relation_embd(relations)
        node_repr, rel_repr = self.multi_layer_comp_gcn(
            concept_repr, 
            rel_repr,
            head_ids,
            tail_ids,
            layer_number=self.num_hops
        )
        head_repr = torch.gather(node_repr, 1, head_ids.unsqueeze(-1).expand(node_repr.size(0), head_ids.size(1), node_repr.size(-1)))
        tail_repr = torch.gather(node_repr, 1, tail_ids.unsqueeze(-1).expand(node_repr.size(0), tail_ids.size(1), node_repr.size(-1)))
        triple_repr = torch.cat((head_repr, rel_repr, tail_repr), dim=-1)

        logging.debug("\tSize of encoded triples: {}".format(triple_repr.size()))

        return triple_repr

    def multi_layer_comp_gcn(self, concept_repr, rel_repr, head_ids, tail_ids, layer_number=2):
        for i in range(layer_number):
            concept_hidden, relation_hidden = self.comp_gcn(
                concept_repr, 
                rel_repr, 
                head_ids, 
                tail_ids, 
                i
            )
        return concept_hidden, relation_hidden

    def comp_gcn(self, concept_repr, rel_repr, head_ids, tail_ids,  layer_idx):
        '''
        concept_repr: B x M x E
        rel_repr: B x Mt x E
        '''
        B = head_ids.size(0)
        Mt = head_ids.size(1)
        M = concept_repr.size(1)
        E = concept_repr.size(2)

        update_node = torch.zeros_like(concept_repr).to(concept_repr.device).float()
        count = torch.ones_like(head_ids).to(head_ids.device).float()
        count_out = torch.zeros(B, M).to(head_ids.device).float()

        o = concept_repr.gather(1, head_ids.unsqueeze(2).expand(B, Mt, E))
        scatter_add(o, tail_ids, dim=1, out=update_node)
        scatter_add(count, tail_ids, dim=1, out=count_out)

        o = concept_repr.gather(1, tail_ids.unsqueeze(2).expand(B, Mt, E))
        scatter_add(o, head_ids, dim=1, out=update_node)
        scatter_add(count, head_ids, dim=1, out=count_out)

        act = nn.ReLU()
        update_node = \
            self.W_s[layer_idx](concept_repr) + \
            self.W_n[layer_idx](update_node) / count_out.clamp(min=1).unsqueeze(2)
        update_node = act(update_node)

        return update_node, self.W_r[layer_idx](rel_repr)


class KnowledgeGroundedDecoder(nn.Module):

    def __init__(self, opt, dict):
        super().__init__()
        self.num_hops = opt['num_hops']
        self.gamma = opt['gamma']
        self.aggregate_method = opt['aggregate_method']

        self.gpt2model = GPT2Decoder(opt, dict)
        self.lm_head = nn.Linear(opt['embedding_size'], len(dict.keys()), bias=False)
        self.lm_head.weight = self.gpt2model.transformer.wte.weight

        self.triple_encoder = TripleEncoder(self.gpt2model.transformer.wte, self.num_hops, opt['embedding_size'])
        self.triple_linear = nn.Linear(opt['embedding_size'] * 3, opt['embedding_size'], bias=False)
        self.gate_linear = nn.Linear(opt['embedding_size'], 1)

        logging.info("Initialized KnowledgeGroundedDecoder")


    def forward(self, decoder_input, encoder_state, incr_state=None):

        logging.debug("Forward KnowledgeGroundedDecoder")

        if incr_state is None:
            input_ids, concept_ids, relations, head_ids, tail_ids, vocab_map, map_mask = encoder_state
            gpt_states = None
            triple_repr = self.triple_encoder(concept_ids, relations, head_ids, tail_ids)
        else:
            input_ids, gpt_states, concept_ids, relations, triple_repr, head_ids, tail_ids, vocab_map, map_mask = incr_state

        probs, gpt_states = self.knowledge_grounded_probs(
            decoder_input, 
            input_ids,
            gpt_states,
            memory_dict={
                "concepts": concept_ids,
                "relations": relations,
                "triple_repr": triple_repr,
                "head": head_ids,
                "tail": tail_ids,
                "vocab_map": vocab_map,
                "map_mask": map_mask
            }
        )
        return probs, (input_ids, gpt_states, concept_ids, relations, triple_repr, head_ids, tail_ids, vocab_map, map_mask)


    def knowledge_grounded_probs(self, decoder_input, input_ids, gpt_states, memory_dict):
        '''
        return: 
            - probs: bsz x L x vocab
            - gate: bsz x L x 1
        '''

        sigmoid = nn.Sigmoid()
        softmax = nn.Softmax(dim=-1)

        # Calculate probabilities according to language model
        hidden_states, gpt_states = self.gpt2model(decoder_input, input_ids, gpt_states)
        lm_logits = self.lm_head(hidden_states)
        lm_probs = softmax(lm_logits)

        # Combine hidden states with knowledge triples to calculate triple score
        triple_logits = torch.matmul(
            hidden_states, 
            self.triple_linear(memory_dict["triple_repr"]).transpose(1, 2)
        )
        triple_prob = sigmoid(triple_logits)
        # bsz x L x mem_t

        # Aggregate probability to nodes
        unorm_cpt_probs = self.multi_hop(
            memory_dict['concepts'],
            triple_prob, 
            memory_dict["head"], 
            memory_dict["tail"]
        )

        # Calculate probability for concepts
        cpt_probs = softmax(unorm_cpt_probs)
        index = memory_dict["vocab_map"].unsqueeze(0).unsqueeze(0).expand(cpt_probs.size(0), cpt_probs.size(1), -1)
        cpt_probs_vocab = cpt_probs.gather(2, index)
        mask = (memory_dict["map_mask"] == 0).unsqueeze(0).unsqueeze(0)
        cpt_probs_vocab.masked_fill_(mask, 0)

        # Determine gate value (which determines whether to take token from language model or select a concept)
        gate = sigmoid(self.gate_linear(hidden_states))

        probs = gate * cpt_probs_vocab + (1 - gate) * lm_probs 

        return probs, gpt_states


    def multi_hop(self, concepts, triple_prob, head, tail):
        '''
        triple_prob: bsz x L x mem_t
        distance: bsz x mem
        head, tail: bsz x mem_t
        concept_label: bsz x mem
        triple_label: bsz x mem_t

        Init binary vector with source concept == 1 and others 0
        expand to size: bsz x L x mem
        '''
        concept_probs = []
        concept_size = (triple_prob.size(0), triple_prob.size(1), concepts.size(1))
        init_mask = torch.ones_like(concepts).unsqueeze(1).expand(*concept_size).to(concepts.device).float()
        concept_probs.append(init_mask)

        head = head.unsqueeze(1).expand(triple_prob.size(0), triple_prob.size(1), -1)
        tail = tail.unsqueeze(1).expand(triple_prob.size(0), triple_prob.size(1), -1)

        for step in range(self.num_hops):
            '''
            Calculate triple head score
            '''
            node_score = concept_probs[-1]

            triple_head_score = node_score.gather(2, head)
            '''
            Method: 
                - avg:
                    s(v) = Avg_{u \in N(v)} gamma * s(u) + R(u->v) 
                - max: 
                    s(v) = max_{u \in N(v)} gamma * s(u) + R(u->v)
            '''
            update_value = triple_head_score * self.gamma + triple_prob
            out = torch.zeros_like(node_score).to(node_score.device).float()
            if self.aggregate_method == "max":
                scatter_max(update_value, tail, dim=-1, out=out)
            elif self.aggregate_method == "avg":
                scatter_mean(update_value, tail, dim=-1, out=out)           
            concept_probs.append(out)
        
        '''
        Natural decay of concept that is multi-hop away from source
        '''
        total_concept_prob = torch.zeros_like(node_score)
        for prob in concept_probs[1:]:
            total_concept_prob += prob
        # bsz x L x mem
        return total_concept_prob


class KnowledgeGroundedModel(TorchGeneratorModel):

    def __init__(self, opt, dict):
        self.add_start_token = opt["add_start_token"]
        super().__init__(*self._get_special_tokens(opt, dict))

        self.encoder = Identity()
        self.decoder = KnowledgeGroundedDecoder(opt, dict)

    def _get_special_tokens(self, opt, dict):
        return dict.null_idx, dict.start_idx, dict.end_idx

    def reorder_encoder_states(self, encoder_states, indices):
        input_ids, concept_ids, relations, head_ids, tail_ids, vocab_map, map_mask = encoder_states
        return (input_ids[:, indices], concept_ids, relations, head_ids, tail_ids, vocab_map, map_mask)


    def reorder_decoder_incremental_state(self, incr_state, indices):

        input_ids, gpt_states, concept_ids, relations, triple_repr, head_ids, tail_ids, vocab_map, map_mask = incr_state
        new_gpt_state = []
        for layer_past in gpt_states:
            if torch.is_tensor(layer_past):
                new_gpt_state.append(torch.index_select(layer_past, 1, indices))
            else:
                # newer versions of HF split up the intermediate outputs
                assert isinstance(layer_past, tuple)
                layer_past = torch.stack(layer_past, dim=0)
                new_gpt_state.append(torch.index_select(layer_past, 1, indices))

        return (input_ids[:, indices], tuple(new_gpt_state), concept_ids, relations, triple_repr, head_ids, tail_ids, vocab_map, map_mask)


    def output(self, decoder_output):
        return decoder_output


    def decode_forced(self, encoder_states, ys):
        """
        Override to get rid of start token input.
        """
        if self.add_start_token:
            return super().decode_forced(encoder_states, ys)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        latent, _ = self.decoder(inputs, encoder_states)
        logits = self.output(latent)
        _, preds = logits.max(dim=2)
        return logits, preds


