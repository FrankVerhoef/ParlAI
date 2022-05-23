import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_max, scatter_mean, scatter_add
from parlai.agents.hugging_face.gpt2 import GPT2Decoder
from parlai.core.torch_generator_agent import TorchGeneratorModel
import parlai.utils.logging as logging


class Identity(nn.Module):
    def forward(self, *batch):
        return tuple(batch)


class TripleEncoder(nn.Module):

    def __init__(self, embedding, num_hops):
        super().__init__()

        E = embedding.weight.shape[-1]
        self.num_hops = num_hops
        self.concept_embd = embedding
        self.relation_embd = nn.Embedding(40, E)
        self.W_s = nn.ModuleList([nn.Linear(E, E, bias=False) for _ in range(self.num_hops)]) 
        self.W_n = nn.ModuleList([nn.Linear(E, E, bias=False) for _ in range(self.num_hops)]) 
        self.W_r = nn.ModuleList([nn.Linear(E, E, bias=False) for _ in range(self.num_hops)])
        self.act = nn.ReLU()

        logging.info("Initialized TripleEncoder")


    def forward(self, concept_ids, relations, head_ids, tail_ids):
        """
        Encodes knowledge triples
        Tensor sizes are:
            B, M: for concept_ids
            B, Mt: for relations, head_ids, tail_ids
            B, Mt, 3 x E: for output (triple representation)

            B = batch size
            M = number of related concepts (can vary per batch)
            Mt = number of related triples (can vary per batch)
            E = embedding dimension for concepts (is same as embedding dim for relations)
        """
        logging.debug("Forward TripleEncoder")
        logging.debug("\tEncoding {} concepts and {} relations".format(concept_ids.size(1), relations.size(1)))

        # Embed concepts and relations
        concept_repr = self.concept_embd(concept_ids)
        rel_repr = self.relation_embd(relations)

        # Calculate GCN representations for concepts and relations, using 'num_hops' layers
        node_repr, rel_repr = self.comp_gcn(
            concept_repr, 
            rel_repr,
            head_ids,
            tail_ids,
            layer_number=self.num_hops
        )

        # Construct triple representation
        head_repr = torch.gather(node_repr, 1, head_ids.unsqueeze(-1).expand(node_repr.size(0), head_ids.size(1), node_repr.size(-1)))
        tail_repr = torch.gather(node_repr, 1, tail_ids.unsqueeze(-1).expand(node_repr.size(0), tail_ids.size(1), node_repr.size(-1)))
        triple_repr = torch.cat((head_repr, rel_repr, tail_repr), dim=-1)

        logging.debug("\tSize of encoded triples: {}".format(triple_repr.size()))

        return triple_repr


    def comp_gcn(self, concept_repr, rel_repr, head_ids, tail_ids, layer_number=2):
        '''
        concept_repr: B x M x E  (M=max number of related concepts)
        rel_repr: B x Mt x E (Mt=max number of triples)
        TODO: implement masking in case batch size != 1
        '''

        B = head_ids.size(0)
        Mt = head_ids.size(1)
        M = concept_repr.size(1)
        E = concept_repr.size(2)

        concept_hidden, relation_hidden = concept_repr, rel_repr
        for l in range(layer_number):

            # Initialise update_node for GCN calculation
            update_node = torch.zeros_like(concept_repr).to(concept_repr.device).float()
            count = torch.ones_like(head_ids).to(head_ids.device).float()
            count_out = torch.zeros(B, M).to(head_ids.device).float()

            # Add the concept representations of the heads to node 'positions' of tails, subtract relation representation
            o = concept_repr.gather(1, head_ids.unsqueeze(2).expand(B, Mt, E))
            scatter_add(o, tail_ids, dim=1, out=update_node)
            scatter_add(-rel_repr, tail_ids, dim=1, out=update_node)
            scatter_add(count, tail_ids, dim=1, out=count_out)

            # Add the concept representations of the tails to node 'position' of heads, subtract relation representation
            o = concept_repr.gather(1, tail_ids.unsqueeze(2).expand(B, Mt, E))
            scatter_add(o, head_ids, dim=1, out=update_node)
            scatter_add(-rel_repr, head_ids, dim=1, out=update_node)
            scatter_add(count, head_ids, dim=1, out=count_out)

            # Combine calculated update to form new node and relation representations
            update_node = \
                self.W_s[l](concept_repr) + \
                self.W_n[l](update_node) / count_out.clamp(min=1).unsqueeze(2)
            concept_hidden = self.act(update_node)
            relation_hidden = self.W_r[l](rel_repr)

        return concept_hidden, relation_hidden


class KnowledgeGroundedDecoder(nn.Module):

    def __init__(self, opt, dict):
        super().__init__()

        # Model and parameters for language model
        self.gpt2model = GPT2Decoder(opt, dict)
        self.lm_head = nn.Linear(opt['embedding_size'], len(dict.keys()), bias=False)
        self.lm_head.weight = self.gpt2model.transformer.wte.weight

        # Model and parameters for knowledge model
        self.num_hops = opt['num_hops']
        self.gamma = opt['gamma']
        self.aggregate_method = opt['aggregate_method']
        self.triple_encoder = TripleEncoder(self.gpt2model.transformer.wte, self.num_hops)
        self.triple_linear = nn.Linear(opt['embedding_size'] * 3, opt['embedding_size'], bias=False)

        # Gate to control generation via language model or knowledge model
        self.gate_linear = nn.Linear(opt['embedding_size'], 1)

        logging.info("Initialized KnowledgeGroundedDecoder")


    def forward(self, decoder_input, encoder_state, incr_state=None):

        #logging.debug("Forward KnowledgeGroundedDecoder")

        if incr_state is None:
            (
                input_ids, 
                concept_ids, distances, relation_ids, head_idx, tail_idx, triple_labels, gate_labels, 
                vocab_map, map_mask 
            ) = encoder_state
            gpt_states = None
            triple_repr = self.triple_encoder(concept_ids, relation_ids, head_idx, tail_idx)
        else:
            (
                input_ids, gpt_states, 
                distances, triple_repr, head_idx, tail_idx, triple_labels, gate_labels, 
                vocab_map, map_mask, triple_prob, gate
            ) = incr_state

        (probs, gate, triple_prob), gpt_states = self.knowledge_grounded_probs(
            decoder_input, 
            input_ids,
            gpt_states,
            memory_dict={
                "distances": distances,
                "triple_repr": triple_repr,
                "head": head_idx,
                "tail": tail_idx,
                "triple_labels": triple_labels,
                "vocab_map": vocab_map,
                "map_mask": map_mask
            }
        )

        incr_state = (
            input_ids, gpt_states, 
            distances, triple_repr, head_idx, tail_idx, triple_labels, gate_labels,
            vocab_map, map_mask, triple_prob, gate
        )

        return probs, incr_state


    def knowledge_grounded_probs(self, decoder_input, input_ids, gpt_states, memory_dict):
        '''
        return: 
            - probs: B x L x V
            - gate: B x L x 1
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
        invalid_mask = (memory_dict["triple_labels"] == -1).unsqueeze(1)
        triple_prob = triple_prob.masked_fill(invalid_mask, 0)
        # B x L x Mt

        # Aggregate probability to nodes
        concept_scores = self.multi_hop(
            triple_prob, 
            memory_dict["distances"],
            memory_dict["head"], 
            memory_dict["tail"]
        )
        concept_probs = softmax(concept_scores)

        # Calculate probability for concepts
        index = memory_dict["vocab_map"].unsqueeze(0).unsqueeze(0).expand(concept_probs.size(0), concept_probs.size(1), -1)
        concept_probs_vocab = concept_probs.gather(2, index)
        invalid_mask = (memory_dict["map_mask"] == 0).unsqueeze(0).unsqueeze(0)
        concept_probs_vocab.masked_fill_(invalid_mask, 0)

        # Determine gate value (which determines whether to take token from language model or select a concept)
        gate = sigmoid(self.gate_linear(hidden_states))

        probs = gate * concept_probs_vocab + (1 - gate) * lm_probs 

        return (probs, gate, triple_prob), gpt_states


    def multi_hop(self, triple_prob, distance, head, tail):
        '''
        triple_prob: B x L x Mt
        distance: B x M
        head, tail: B x Mt
        concept_label: B x M
        triple_label: B x Mt
        '''
        
        # Init binary vector with source concept == 1 and others 0, and expand to size B, L, M
        concept_scores = []
        concept_size = (triple_prob.size(0), triple_prob.size(1), distance.size(1))
        init_mask = torch.zeros_like(distance).unsqueeze(1).expand(*concept_size).to(distance.device).float()
        init_mask.masked_fill_((distance == 0).unsqueeze(1), 1)
        concept_scores.append(init_mask)

        head = head.unsqueeze(1).expand(triple_prob.size(0), triple_prob.size(1), -1)
        tail = tail.unsqueeze(1).expand(triple_prob.size(0), triple_prob.size(1), -1)

        for _ in range(self.num_hops):

            # Calculate triple head score
            node_score = concept_scores[-1]
            triple_head_score = node_score.gather(2, head)
            
            # Aggregate scores to tail nodes
            # avg: score(tail) = avg_{head \in N(tail)} gamma * score(head) + p(head -> tail) 
            # max: score(tail) = max_{head \in N(tail)} gamma * score(head) + p(head -> tail)
            update_value = triple_head_score * self.gamma + triple_prob
            out = torch.zeros_like(node_score).to(node_score.device).float()
            if self.aggregate_method == "max":
                scatter_max(update_value, tail, dim=-1, out=out)
            elif self.aggregate_method == "avg":
                scatter_mean(update_value, tail, dim=-1, out=out)           
            concept_scores.append(out)
        
        # Assign large negative value to start-nodes
        # Apply decay factor for concept scores that are further away from source
        total_concept_score = init_mask * -1e5 # torch.zeros_like(node_score)
        for score in concept_scores[1:]:
            total_concept_score += score * 0.8 ** distance
        # B x L x M

        return total_concept_score


class KnowledgeGroundedModel(TorchGeneratorModel):

    def __init__(self, opt, dict):
        self.add_start_token = opt["add_start_token"]
        super().__init__(*self._get_special_tokens(opt, dict))

        self.encoder = Identity()
        self.decoder = KnowledgeGroundedDecoder(opt, dict)

    def _get_special_tokens(self, opt, dict):
        return dict.null_idx, dict.start_idx, dict.end_idx



    def forward(self, *xs, ys=None, prev_enc=None, maxlen=None, bsz=None):
        """
        Get output predictions from the model.
        """
        assert ys is not None, "Greedy decoding in TGModel.forward no longer supported."

        self.longest_label = max(self.longest_label, ys.size(1))

        # use cached encoding if available
        encoder_states = prev_enc if prev_enc is not None else self.encoder(*xs)

        # use teacher forcing
        scores, preds, triple_probs, gate = self.decode_forced(encoder_states, ys)
        return scores, preds, encoder_states, triple_probs, gate
        

    def reorder_encoder_states(self, encoder_states, indices):
        input_ids, *other_state_values = encoder_states
        return (input_ids[:, indices], *other_state_values)


    def reorder_decoder_incremental_state(self, incr_state, indices):

        input_ids, gpt_states, *other_state_values = incr_state
        new_gpt_state = []
        for layer_past in gpt_states:
            if torch.is_tensor(layer_past):
                new_gpt_state.append(torch.index_select(layer_past, 1, indices))
            else:
                # newer versions of HF split up the intermediate outputs
                assert isinstance(layer_past, tuple)
                layer_past = torch.stack(layer_past, dim=0)
                new_gpt_state.append(torch.index_select(layer_past, 1, indices))

        return (input_ids[:, indices], tuple(new_gpt_state), *other_state_values)


    def output(self, decoder_output):
        probs = decoder_output
        return probs


    def decode_forced(self, encoder_states, ys):
        """
        Override to get rid of start token input.
        """
        if self.add_start_token:
            return super().decode_forced(encoder_states, ys)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        latent, final_state = self.decoder(inputs, encoder_states)
        (
            input_ids, gpt_states, 
            distances, triple_repr, head_ids, tail_ids, triple_labels, gate_labels, 
            vocab_map, map_mask, triple_prob, gate
        ) = final_state
        logits = self.output(latent)
        _, preds = logits.max(dim=2)
        return logits, preds, triple_prob, gate


