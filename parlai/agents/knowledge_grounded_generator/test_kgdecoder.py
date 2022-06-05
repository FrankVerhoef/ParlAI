import torch
import torch.nn as nn

from parlai.agents.knowledge_grounded_generator.kg_model import TripleEncoder
from parlai.agents.knowledge_grounded_generator.knowledge_grounded_generator import KnowledgeGroundedGeneratorAgent

e = torch.tensor(range(10)).unsqueeze(dim=1).expand(10,768).float()
embedding = nn.Embedding.from_pretrained(e)
num_hops=2

triple_encoder = TripleEncoder(embedding=embedding, num_hops=num_hops)

# Change inner parameters
r = torch.tensor(range(40)).unsqueeze(dim=1).expand(40,768).float()
triple_encoder.relation_embd = nn.Embedding.from_pretrained(r)
triple_encoder.W_s = nn.ModuleList([nn.Identity() for _ in range(num_hops)]) 
triple_encoder.W_r = nn.ModuleList([nn.Identity() for _ in range(num_hops)]) 
triple_encoder.W_n = nn.ModuleList([nn.Identity() for _ in range(num_hops)]) 

c2i = {"A":0, "F":1, "G": 2, "N": 3, "B": 4, "H":5, "C":6, "E":7, "D":8, "K":9}
i2c = ["A", "F", "G", "N", "B", "H", "C", "E", "D", "K"]

concept_ids = torch.tensor([[0, 1, 4, 5, 7, 2, 8, 6]])
relations = torch.tensor([[1, 1, 2, 3, 1, 1, 2, 3, 3]])
head_ids = torch.tensor([[0, 0, 0, 0, 1, 4, 4, 3, 2]])
tail_ids = torch.tensor([[1, 4, 3, 2, 5, 6, 3, 7, 7]])

encoding = triple_encoder(concept_ids, relations, head_ids, tail_ids)
#print(encoding)

opt = {
    "num_hops": 2,
    "aggregate_method": "max",
    "embedding_size": 768,
    "max_concepts": 400,
    "max_triples": 800,
    "alpha": 0.7,
    "beta": 0.2,
    "gamma": 0.33,
    'concepts': '/users/FrankVerhoef/Programming/Project_AI/multigen/data/concept.txt', 
    'relations': '/users/FrankVerhoef/Programming/Project_AI/multigen/data/relation.txt', 
    'dataset_concepts': '/users/FrankVerhoef/Programming/Project_AI/multigen/data/anlg/total_concepts.txt',
    'kg': '/users/FrankVerhoef/Programming/Project_AI/multigen/data/cpnet_25.graph', 
    'hidden_size': 256,
    "source_length": 16,
    "max_memory_size": 400,

    "add_special_tokens": True,
    "add_start_token": False,
    "no_cuda": True,
    "model_name": None,
    "gpt2_size": "small",
    "datapath": "data/",
    "history_size": 16,
    "truncate": 64,
    "rank_candidates": False,
    "gate": None
}

agent = KnowledgeGroundedGeneratorAgent(opt)


# agent.model.decoder.triple_encoder = triple_encoder
obs = {
    "text": "I like my mother and sister.",
    "eval_labels": ["Your family is important since birth"],
}
related_concepts = ["A", "F", "B", "H", "E", "G", "D", "C", "N", "K"]
agent.observe(obs)
# agent.observation.force_set('related_concepts', related_concepts)
# agent.observation.force_set('concept_token_ids', torch.LongTensor([c2i[c] for c in related_concepts]))
# agent.observation.force_set('concept_labels', torch.LongTensor([1, 0, 0, 1, 1, 1, 0, 1, 0, 0]))
# agent.observation.force_set('distances', torch.LongTensor([0, 1, 1, 1, 1, 2, 2, 2, 3, 3]))
# agent.observation.force_set('relation_ids', torch.LongTensor(relations[0]))
# agent.observation.force_set('head_idx', torch.LongTensor(head_ids[0]))
# agent.observation.force_set('tail_idx', torch.LongTensor(tail_ids[0]))
# agent.observation.force_set('triple_labels', torch.ones_like(head_ids[0]))
# agent.observation.force_set('gate_labels', torch.LongTensor([-1, 0, 1, 0, 0, 0, 1]))
agent.observation.force_set('episode_done', False)

reply = agent.act()
print(reply)