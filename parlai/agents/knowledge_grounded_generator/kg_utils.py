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

# Blacklist contains concepts NOT to include.
# In this case, it is the list with auxiliary verbs and the word 'persona' 
# which is used as identifier in the input text for the persona descriptions.
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
        """
        
        cpnet = pickle.load(open(graph_path, "rb"))
        self.graph = cpnet
        logging.info("Loaded knowledge graph with {} nodes and {} edges".format(len(cpnet.nodes), len(cpnet.edges)))


    def build_reduced_graph(self, concepts_subset):
        """
            Construct a subgraph with only the nodes in the given concepts_subset.
            In this graph multiple edges between the same nodes are combined (and weight added)
        """
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
        result = set()

        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN', 'VERB']:
                if token.text in self.simple_vocab:
                    result.add(token.text)

        return result


    def match_mentioned_concepts(self, source, target, overlapping_concepts):
        """
            Returns a dict with concepts from the input sentence and the target sentence
        """

        if overlapping_concepts == "keep-src-and-tgt":
            source_concepts = self.hard_ground(source)
            target_concepts = self.hard_ground(target)            
        else:
            combined_concepts = self.hard_ground(source + ' ' + target)
            if overlapping_concepts == "excl-tgt-in-src":
                target_concepts = self.hard_ground(target)
                source_concepts = combined_concepts - target_concepts
            else: # excl-src-in-tgt
                source_concepts = self.hard_ground(source)
                target_concepts = combined_concepts - source_concepts
        return {"source_concepts": list(source_concepts), "target_concepts": list(target_concepts)}


    def get_relations(self, src_concept, tgt_concept):
            
        try:
            rel_list = self.graph[src_concept][tgt_concept]
            return list(set([rel_list[item]["rel"] for item in rel_list]))
        except:
            return []


    def find_neighbours_nx(self, source_concepts, target_concepts, num_hops, max_B=100):
        """
            Find neighboring concepts within num_hops and the connecting triples
            Use graph functions from networkx
        """
        # id's in knowledge graph of the source and target concepts   
        source_ids = set(self.concept2id[s_cpt] for s_cpt in source_concepts)
        target_ids = [self.concept2id[t_cpt] for t_cpt in target_concepts]
        all_concepts = self.simple_graph.nodes

        # Vts init contains id's of source concepts with distance 0
        Vts = dict([(x,0) for x in source_ids])
        Ets = {}

        related = source_ids
        current_boundary = source_ids
        for t in range(num_hops):
            V = {}
            for v in nx.node_boundary(self.simple_graph, current_boundary, all_concepts - related):
                incoming_nodes = list(nx.node_boundary(self.simple_graph, [v], current_boundary))
                V[v] = sum(
                    self.simple_graph[u][v].get('weight', 1) 
                    for u in incoming_nodes
                )
                Ets.update(dict([(v, dict([
                    (u, self.get_relations(u, v))
                    for u in incoming_nodes
                ]))]))

            # Select nodes that are 'most' connected
            top_V = sorted(list(V.items()), key=lambda x: x[1], reverse=True)[:max_B]
            new_boundary = [v[0] for v in top_V]

            # Add nodes to Vts, with distance increased by 1
            Vts.update(dict([(v, t+1) for v in new_boundary]))
            related.update(new_boundary)
            current_boundary = new_boundary

        concept_ids = [id for id in Vts.keys()]
        labels = [int(c in target_ids) for c in concept_ids]
        distances = [d for d in Vts.values()]
        triples = [
            (u, rels, v)
            for v, incoming_relations in Ets.items()
            for u, rels in incoming_relations.items()
            if (u in concept_ids) and (v in concept_ids)
        ]

        return {"concept_ids":concept_ids, "labels":labels, "distances":distances, "triples":triples}


    def filter_directed_triple(self, related_concepts, max_concepts=64, max_triples=256):

        num_concepts = len(related_concepts['concept_ids'])
        if num_concepts > max_concepts:
            logging.warning("Number of connected concepts {} larger than max-concepts {}. If this happens frequently, consider to increase max-concepts".format(
                num_concepts, max_concepts
            ))
            num_concepts = max_concepts
        concept_ids = related_concepts['concept_ids'][:max_concepts]
        labels = related_concepts['labels'][:max_concepts]
        distances = related_concepts['distances'][:max_concepts]
        triples = related_concepts['triples']

        # Construct triple_dict, with per tail-node, all the triples that are connected
        triple_dict = {}
        for triple in triples:
            head, _, tail = triple
            try:
                head_index = concept_ids.index(head)
                tail_index = concept_ids.index(tail)
                if distances[head_index] <= distances[tail_index]:
                    if tail not in triple_dict:
                        triple_dict[tail] = [triple]
                    else:
                        triple_dict[tail].append(triple)
            except ValueError:
                # If head or tail not found in concept_ids (because of truncation), just pass
                pass

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

        # Sort triple lists, one list per tail node
        triple_lists_sorted = sorted(list(triple_dict.values()), key=lambda x: len(x), reverse=False)
        num_triples = sum([len(triple_list) for triple_list in triple_lists_sorted])
        if num_triples > max_triples:
            logging.warning("Number of connected concepts {} larger than max-triples {}. If this happens frequently, consider to increase max-triples".format(
                num_triples, max_triples
            ))
            num_triples = max_triples

        # Loop through triple lists. This can never be more than max_triples; rest is truncated
        num_triple_lists = min(max_triples, len(triple_lists_sorted))
        for i, triple_list in enumerate(triple_lists_sorted[:num_triple_lists]):
            max_neighbors = (max_triples - triple_count) // (num_triple_lists - i)
            for (head, rels, tail) in triple_list[:max_neighbors]:
                heads.append(concept_ids.index(head))
                tails.append(concept_ids.index(tail))
                relations.append(rels[0])   # Keep only one relation
                triple_labels.append(int((tail, head) in ground_truth_triples_set))
                triple_count += 1

        logging.debug("Connecting paths: {} with {} triples; kept {} triples with {} targets".format(
            len(shortest_paths), len(ground_truth_triples_set), len(triple_labels), sum(triple_labels)
        ))
        logging.debug("Examples: {}".format([
            " - ".join([self.id2concept[n] for n in p])
            for p in shortest_paths
        ][:5]))

        related_concepts['head_idx'] = heads
        related_concepts['tail_idx'] = tails
        related_concepts['relation_ids'] = relations
        related_concepts['triple_labels'] = triple_labels
            
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

