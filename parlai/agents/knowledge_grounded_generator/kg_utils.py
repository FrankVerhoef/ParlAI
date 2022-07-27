import networkx as nx
import json
import pickle
import spacy
import csv
import parlai.utils.logging as logging

HEAD=0
REL=1
TAIL=2

nlp = spacy.load('en_core_web_sm', disable=['parser', 'lemmatizer'])

blacklist = set([
    "persona", 
    "be", "am", "is", "are", "was", "were", "being", "been",
    "do", "does", "did",
    "can", "could",
    "will", "would", "wo",
    "have", "has", "had",
    "must",
    "shall", "should",
    "may", 
    "might",
    "dare", 
    "need",
    "ought"
])


def read_csv(data_path="train/source.csv"):
    data = []
    with open(data_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            data.append(' '.join(row[1:]))
    return data

def read_json(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def save_json(data, filename):
    with open(filename, 'w') as f:
        for line in data:
            json.dump(line, f)
            f.write('\n')

NOCONCEPT_TOKEN = '<NoConcept>'
NORELATION_TOKEN = '<NoRelation>'

class ConceptGraph(nx.Graph):

    def __init__(self, path, concepts, relations, graph):
        super().__init__()
        self.load_resources(path + concepts, path + relations)
        self.load_knowledge_graph(path + graph)


    def load_resources(self, concepts, relations):

        concept2id = {}
        id2concept = {}
        cpnet_vocab = []
        with open(concepts, "r", encoding="utf8") as f:
            for line in f.readlines():
                w = line.strip()
                concept2id[w] = len(concept2id)
                id2concept[len(id2concept)] = w
                cpnet_vocab.append(w)
        concept2id[NOCONCEPT_TOKEN] = len(concept2id)
        id2concept[len(id2concept)] = NOCONCEPT_TOKEN
        self.concept2id = concept2id
        self.id2concept = id2concept
        self.vocab = set([c.replace("_", " ") for c in cpnet_vocab])
        logging.debug("Loaded {} concepts".format(len(self.concept2id)))

        id2relation = {}
        relation2id = {}
        with open(relations, "r", encoding="utf8") as f:
            for w in f.readlines():
                id2relation[len(id2relation)] = w.strip()
                relation2id[w.strip()] = len(relation2id)
        l = len(relation2id)
        for i in range(len(relation2id)): 
            reverse = 'reverse_' + id2relation[i]
            id2relation[len(id2relation)] = reverse
            relation2id[reverse] = len(relation2id)
        id2relation[len(id2relation)] = NORELATION_TOKEN
        relation2id[NORELATION_TOKEN] = len(relation2id)

        self.id2relation = id2relation
        self.relation2id = relation2id
        logging.debug("Loaded {} relation types".format(len(relation2id)))


    def load_knowledge_graph(self, graph_path):
        """
            Load the graph and store it, as well as a simpler version of the graph.
            In this graph multiple edges between the same nodes are combined (and weight added)
        """
        
        cpnet = pickle.load(open(graph_path, "rb"))
        self.graph = cpnet
        logging.info("Loaded knowledge graph with {} nodes and {} edges".format(len(cpnet.nodes), len(cpnet.edges)))


    def build_reduced_graph(self, concepts_subset):

        matching_ids = [self.concept2id[c] for c in concepts_subset if c in self.concept2id.keys()]
        cpnet_simple = nx.Graph()

        for u, v, data in nx.subgraph(self.graph, matching_ids).edges(data=True):
            w = data['weight'] if 'weight' in data else 1.0
            if cpnet_simple.has_edge(u, v):
                cpnet_simple[u][v]['weight'] += w
            else:
                cpnet_simple.add_edge(u, v, weight=w)

        self.simple_graph = cpnet_simple
        self.simple_vocab = [self.id2concept[u] for u in cpnet_simple.nodes()]

        logging.info("Built reduced graph with {} nodes and {} edges".format(len(cpnet_simple.nodes), len(cpnet_simple.edges)))


    def hard_ground(self, sent):
        """
            Returns a list of verbs and nouns in the input sentence
            that also occur in the (reduced) ConceptNet vocabulary
        """
        # TODO: Need to think about how to match words with concepts in ConceptNet
        # 'father in law' is 1 concept in ConceptNet, but Spacy tokenizer breaks it into three words,
        # 'father', 'in', 'law' which also occur in ConceptNet

        sent = sent.lower()
        doc = nlp(sent)
        res = set()

        # logging.debug("POS tags: {}".format(
        #     ', '.join(['({}, {})'.format(token.text, token.pos_) for token in doc])
        # ))

        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN', 'VERB']:
                if token.text in self.simple_vocab:
                    res.add(token.text)

        return res


    def match_mentioned_concepts(self, sent, answer):
        """
            Returns a dict with sentence from source, sentence from answer and the concepts
            from those sentences
        """

        all_concepts = self.hard_ground(sent + ' ' + answer)
        question_concepts = self.hard_ground(sent)
        answer_concepts = all_concepts - question_concepts
        return {"sent": sent, "ans": answer, "qc": list(question_concepts), "ac": list(answer_concepts)}


    def get_edge(self, src_concept, tgt_concept):
            
        try:
            rel_list = self.graph[src_concept][tgt_concept]
            return list(set([rel_list[item]["rel"] for item in rel_list]))
        except:
            return []


    def find_neighbours(self, source_concepts, target_concepts, num_hops, max_B=100):
        """
            Find neighboring concepts within num_hops and the connecting triples
        """
        # logging.debug("Finding neighbours for {} and {}".format(source_concepts, target_concepts))
        source = [self.concept2id[s_cpt] for s_cpt in source_concepts]  # id's in knowledge graph of the source concepts 
        start = source                              # start init contains id's of source concepts
        Vts = dict([(x,0) for x in start])          # Vts init contains id's of concepts in knowledge graph, with distance 0
        Ets = {}
        for t in range(num_hops):
            V = {}
            for s in start:
                # logging.debug("Check neighbors of: {}".format(self.id2concept[s]))
                for n in self.simple_graph[s]:      # loops through the neighbors
                    if n not in Vts:
                        if n not in V:              # if not yet reached, add node to 'V' with weight
                            V[n] = self.simple_graph[s][n]['weight']
                        else:                       # if already reached, increase weight of node
                            V[n] += self.simple_graph[s][n]['weight'] 
                    rels = self.get_edge(s, n)      # list of relation types between s and n in the full graph
                    if len(rels) > 0:
                        if n not in Ets:
                            Ets[n] = {s: rels}  
                        else:
                            Ets[n].update({s: rels})
                            
            V = list(V.items())                     # convert from dict to list
            count_V = sorted(V, key=lambda x: x[1], reverse=True)[:max_B] # select concepts with most weight
            start = [x[0] for x in count_V]         # update start to the newly visited nodes
            
            # Add nodes to Vts, with distance increased by 1
            Vts.update(dict([(x, t+1) for x in start]))
        
        concept_ids = [id for id in Vts.keys()]
        distances = [d for d in Vts.values()]

        # Construct tuples with triples
        triples = []
        for v, N in Ets.items():
            if v in concept_ids:
                for u, rels in N.items():
                    if u in concept_ids:
                        triples.append((u, rels, v))

        # id's of nodes in the concept graph of target concepts
        target_ids = [self.concept2id[t_cpt] for t_cpt in target_concepts]   

        # Construct a list with labels; if the T-hop concept appears in target, then corresponding label is 1
        labels = [int(c in target_ids) for c in concept_ids]
        
        # Translate concept id's and relation id's back to text for interpretability
        # concepts = [self.id2concept[c] for c in concept_ids] 
        # triples_text = [(self.id2concept[u], [self.id2relation[r] for r in rels], self.id2concept[v]) for (u, rels, v) in triples]
        # logging.debug("\tReturn {} concepts and {} triples".format(len(concepts), len(triples)))

        return {"concept_ids":concept_ids, "labels":labels, "distances":distances, "triples":triples}, sum(labels), len(concept_ids)


    def filter_directed_triple(self, related_concepts, max_concepts=64, max_triples=256, max_neighbors=8):

        concept_ids = related_concepts['concept_ids']
        labels = related_concepts['labels']
        distances = related_concepts['distances']
        triples = related_concepts['triples']

        triple_dict = {}
        for triple in triples:
            head, _, tail = triple
            head_index = concept_ids.index(head)
            tail_index = concept_ids.index(tail)
            if distances[head_index] <= distances[tail_index]:
                if tail not in triple_dict:
                    triple_dict[tail] = [triple]
                else:
                    if len(triple_dict[tail]) < max_neighbors:
                        triple_dict[tail].append(triple)

        targets = [id for id, l in zip(concept_ids, labels) if l == 1]
        sources = [id for id, d in zip(concept_ids, distances) if d == 0]
        shortest_paths = []
        for target in targets:
            shortest_paths.extend(bfs(target, triple_dict, sources))

        ground_truth_triples_set = set([
            (n, path[i+1]) 
            for path in shortest_paths 
            for i, n in enumerate(path[:-1])
        ])

        heads, tails, relations, triple_labels = [], [], [], []
        triple_count = 0
        for triple_list in triple_dict.values():
            for (head, rels, tail) in triple_list:
                max_reached = triple_count >= max_triples
                if max_reached: break
                heads.append(concept_ids.index(head))
                tails.append(concept_ids.index(tail))
                relations.append(rels[0])   # Keep only one relation
                triple_labels.append(int((tail, head) in ground_truth_triples_set))
                triple_count += 1
            if max_reached: break

        logging.debug("Connecting paths: {} with {} triples; kept {} triples with {} targets".format(
            len(shortest_paths), len(ground_truth_triples_set), len(triple_labels), sum(triple_labels)
        ))
        logging.debug("Examples: {}".format([
            " + ".join([self.id2concept[n] for n in p])
            for p in shortest_paths
        ][:5]))

        related_concepts['head_idx'] = heads
        related_concepts['tail_idx'] = tails
        related_concepts['relation_ids'] = relations
        related_concepts['triple_labels'] = triple_labels
        related_concepts.pop('triples')
            
        return related_concepts


    def formatted_concepts_string(self, related_concepts, max):
        concepts = [self.id2concept[id] for id in related_concepts['concept_ids']]
        if len(concepts) > max and max > 4:
            return 'Examples: ' + ', '.join(concepts[:max//2]) + ' ... ' + ', '.join(concepts[-max//2:])
        else:
            return(', '.join(concepts))
 

    def formatted_triples_string(self, related_concepts, max):
        n = max if len(related_concepts['relation_ids']) > max else len(related_concepts['relation_ids'])
        return ', '.join([
            '({}, {}, {}) '.format(
                self.id2concept[related_concepts['concept_ids'][related_concepts['head_idx'][i]]],
                self.id2relation[related_concepts['relation_ids'][i]],
                self.id2concept[related_concepts['concept_ids'][related_concepts['tail_idx'][i]]]
            )
            for i in range(n)
        ])


def bfs(target, triple_dict, sources, max_steps=2):
    """
        Perform breath-first-search and return all paths that connect nodes in sources to target
    """
    paths = [[[target]]]
    connecting_paths = []
    for _ in range(max_steps):
        last_paths = paths[-1]
        new_paths = []
        for path in last_paths:
            for triple in triple_dict.get(path[-1], []):
                new_paths.append(path + [triple[HEAD]])

        for path in new_paths:
            if path[-1] in sources:
                connecting_paths.append(path)
        
        paths.append(new_paths)
    
    return connecting_paths

