import torch
import torch.nn as nn

from parlai.agents.knowledge_grounded_generator.kg_model import TripleEncoder

e = torch.tensor(range(10)).unsqueeze(dim=1).expand(10,8).float()
embedding = nn.Embedding.from_pretrained(e)
num_hops=2

triple_encoder = TripleEncoder(embedding=embedding, num_hops=num_hops)

# Change inner parameters
r = torch.tensor(range(40)).unsqueeze(dim=1).expand(40,8).float()
triple_encoder.relation_embd = nn.Embedding.from_pretrained(r)
triple_encoder.W_s = nn.ModuleList([nn.Identity() for _ in range(num_hops)]) 
triple_encoder.W_r = nn.ModuleList([nn.Identity() for _ in range(num_hops)]) 
triple_encoder.W_n = nn.ModuleList([nn.Identity() for _ in range(num_hops)]) 


concept_ids = torch.tensor([[0, 1, 4, 5, 7, 2, 8, 6]])
relations = torch.tensor([[1, 1, 2, 3, 1, 1, 2, 3, 3]])
head_ids = torch.tensor([[0, 0, 0, 0, 1, 4, 4, 3, 2]])
tail_ids = torch.tensor([[1, 4, 3, 2, 5, 6, 3, 7, 7]])

encoding = triple_encoder(concept_ids, relations, head_ids, tail_ids)
print(encoding)

