import networkx as nx
import json
import pickle
import spacy
import csv
import parlai.utils.logging as logging

nlp = spacy.load('en_core_web_sm')
#nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])

# How has this blacklist been determined ???
blacklist = set([
    "from", "as", "more", "either", "in", "and", "on", "an", "when", "too", "to", "i", "do", "can", "be", "that", 
    "or", "the", "a", "of", "for", "is", "was", "the", "-PRON-", "actually", "likely", "possibly", "want",
    "make", "my", "someone", "sometimes_people", "sometimes","would", "want_to",
    "one", "something", "sometimes", "everybody", "somebody", "could", "could_be","mine","us","em",
    "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "A", "a1", "a2", "a3", "a4", "ab", "able", "about",
    "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", 
    "ad", "added", "adj", "ae", "af", "affected", "affecting", "after", "afterwards", "ag", "again", 
    "against", "ah", "ain", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", 
    "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", 
    "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anyway", 
    "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appreciate", "approximately", "ar", "are", 
    "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "au", "auth", "av", "available", 
    "aw", "away", "awfully", "ax", "ay", "az", "b", "B", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", 
    "became", "been", "before", "beforehand", "beginnings", "behind", "below", "beside", "besides", "best", 
    "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", 
    "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "C", "c1", "c2", "c3", "ca", "call", "came", "can", 
    "cannot", "cant", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "ci", "cit", "cj", "cl", 
    "clearly", "cm", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", 
    "considering", "could", "couldn", "couldnt", "course", "cp", "cq", "cr", "cry", "cs", "ct", "cu", "cv", 
    "cx", "cy", "cz", "d", "D", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", 
    "despite", "detail", "df", "di", "did", "didn", "dj", "dk", "dl", "do", "does", "doesn", "doing", "don", 
    "done", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "E", "e2", "e3", 
    "ea", "each", "ec", "ed", "edu", "ee", "ef", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", 
    "else", "elsewhere", "em", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", 
    "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", 
    "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "F", "f2", "fa", "far", "fc", 
    "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "five", "fix", "fj", "fl", "fn", 
    "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", 
    "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "G", "ga", "gave", 
    "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", 
    "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "H", "h2", "h3", "had", "hadn", "happens", 
    "hardly", "has", "hasn", "hasnt", "have", "haven", "having", "he", "hed", "hello", "help", "hence", "here", 
    "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hh", "hi", "hid", "hither", "hj", "ho", 
    "hopefully", "how", "howbeit", "however", "hr", "hs", "http", "hu", "hundred", "hy", "i2", "i3", "i4", "i6", 
    "i7", "i8", "ia", "ib", "ibid", "ic", "id", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "im", 
    "immediately", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", 
    "information", "inner", "insofar", "instead", "interest", "into", "inward", "io", "ip", "iq", "ir", "is", 
    "isn", "it", "itd", "its", "iv", "ix", "iy", "iz", "j", "J", "jj", "jr", "js", "jt", "ju", "just", "k", "K", 
    "ke", "keep", "keeps", "kept", "kg", "kj", "km", "ko", "l", "L", "l2", "la", "largely", "last", "lately", 
    "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "lf", "like", 
    "liked", "likely", "line", "little", "lj", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", 
    "lt", "ltd", "m", "M", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", 
    "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mill", "million", "mine", "miss", "ml", 
    "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", 
    "must", "mustn", "my", "n", "N", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", 
    "necessarily", "neither", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", 
    "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "novel", 
    "now", "nowhere", "nr", "ns", "nt", "ny", "o", "O", "oa", "ob", "obtain", "obtained", "obviously", "oc", 
    "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", 
    "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "otherwise", "ou", "ought", "our", 
    "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "P", "p1", "p2", "p3", "page", 
    "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", 
    "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "pp", 
    "pq", "pr", "predominantly", "presumably", "previously", "primarily", "probably", "promptly", "proud", 
    "provides", "ps", "pt", "pu", "put", "py", "q", "Q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "R", 
    "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", 
    "refs", "regarding", "regardless", "regards", "related", "relatively", "research-articl", "respectively", 
    "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", 
    "rt", "ru", "run", "rv", "ry", "s", "S", "s2", "sa", "said", "saw", "say", "saying", "says", "sc", "sd", "se", 
    "sec", "second", "secondly", "section", "seem", "seemed", "seeming", "seems", "seen", "sent", "seven", 
    "several", "sf", "shall", "shan", "shed", "shes", "show", "showed", "shown", "showns", "shows", "si", "side", 
    "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somehow", "somethan",
    "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", 
    "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", 
    "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "sz", "t", "T", "t1", "t2", "t3", 
    "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", 
    "thanks", "thanx", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", 
    "thereafter", "thereby", "thered", "therefore", "therein", "thereof", "therere", "theres", "thereto", 
    "thereupon", "these", "they", "theyd", "theyre", "thickv", "thin", "think", "third", "this", "thorough", 
    "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", 
    "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", 
    "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "tt", "tv", "twelve", "twenty", 
    "twice", "two", "tx", "u", "U", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", 
    "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "used", "useful", 
    "usefully", "usefulness", "using", "usually", "ut", "v", "V", "va", "various", "vd", "ve", "very", "via", 
    "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "W", "wa", "was", "wasn", "wasnt", 
    "way", "we", "wed", "welcome", "well", "well-b", "went", "were", "weren", "werent", "what", "whatever", 
    "whats", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", 
    "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", 
    "whom", "whomever", "whos", "whose", "why", "wi", "widely", "with", "within", "without", "wo", "won", 
    "wonder", "wont", "would", "wouldn", "wouldnt", "www", "x", "X", "x1", "x2", "x3", "xf", "xi", "xj", "xk", 
    "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "Y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "your", 
    "youre", "yours", "yr", "ys", "yt", "z", "Z", "zero", "zi", "zz"
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
        logging.info("Loaded knowledge graph with {} nodes and {} edges".format(len(cpnet.nodes), len(cpnet.edges)))

        cpnet_simple = nx.Graph()
        for u, v, data in cpnet.edges(data=True):
            w = data['weight'] if 'weight' in data else 1.0
            if cpnet_simple.has_edge(u, v):
                cpnet_simple[u][v]['weight'] += w
            else:
                cpnet_simple.add_edge(u, v, weight=w)
        logging.debug("Built simplified graph with {} nodes and {} edges".format(len(cpnet_simple.nodes), len(cpnet_simple.edges)))

        self.graph = cpnet
        self.simple_graph = cpnet_simple


    def hard_ground(self, sent):
        """
            Returns a list of (lemmatized) verbs and nouns in the input sentence
            that also occur in the ConceptNet vocabulary and in the model vocabulary
            and are not in the 'blacklist'
            
        """
        sent = sent.lower()
        doc = nlp(sent)
        res = set()
        
        # TODO: Need to think about how to match words with ConceptNet vocab and GPT2 vocab
        # ConceptNet: probably convert to lowercase and 
        # GPT2: Uses BPE tokenizer, so no unknown words. But it does have differences between uppercase/lowercase
        # and also distinguishes between wordparts IN a sentence and words with space before. Examples:
        # tokenize 'read' != ' read'
        # 'father in law' is 1 concept in ConceptNet, but three tokens in GPT2

        # print("Hard ground <{}>".format(sent))

        for c in doc.noun_chunks:
            if c.root.text in self.vocab and c.root.text not in blacklist:
                res.add(c.root.text)
    #            print("added ", c.root.text, c.text)
    #        else:
    #            print("skipped ", c.root.text, c.text)

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


    def find_neighbours(self, source_concepts, target_concepts, dataset_concepts, num_hops, max_B=100):
        """
            Find ...
        """
        logging.debug("Finding neighbours for {} and {}".format(source_concepts, target_concepts))
        source = [self.concept2id[s_cpt] for s_cpt in source_concepts]  # id's in knowledge graph of the source concepts 
        start = source                              # start init contains id's of source concepts
        Vts = dict([(x,0) for x in start])          # Vts init contains id's of concepts in knowledge graph, with distance 0
        Ets = {}
        #dataset_concept2id_set = set(dataset_concept2id)
        for t in range(num_hops):      # T is number of hops
            V = {}
            templates = []
            for s in start:
                if s in self.simple_graph:
                    logging.debug("Check neighbors of: {}".format(self.id2concept[s]))
                    for n in self.simple_graph[s]:       # loops through the neighbors
                        if n not in Vts and self.id2concept[n] in dataset_concepts:
                            if n not in Vts:
                                if n not in V:      # if not yet reached, add to 'V' with frequency
                                    logging.debug("New node: {}".format(self.id2concept[n]))
                                    V[n] = 1
                                else:
                                    V[n] += 1       # if already reached, increase frequency

                            if n not in Ets:
                                rels = self.get_edge(s, n)   # list of relation types between s and n
                                if len(rels) > 0:
                                    Ets[n] = {s: rels}  
                            else:
                                rels = self.get_edge(s, n)
                                if len(rels) > 0:
                                    Ets[n].update({s: rels})
                        # else:
                            # logging.debug("Not added: {}".format(self.id2concept[n]))
                # else:
                    # logging.debug("Not in simple graph: {}".format(self.id2concept[s]))
                            
            V = list(V.items())         # convert from dict to list
            count_V = sorted(V, key=lambda x: x[1], reverse=True)[:max_B] # select most frequently visited
            start = [x[0] for x in count_V if self.id2concept[x[0]] in dataset_concepts] # update start to the newly visited nodes
            
            # Add nodes to Vts, with distance increased by 1
            Vts.update(dict([(x, t+1) for x in start]))

            logging.debug("\tResult after hop {}: \t{} new nodes (max 10): {}".format(
                t, len(V), V[:10]
            ))
        
        # Unclear what the purpose is of these lines. Doesn't seem to change concepts & distances
        _concept_ids = list(Vts.keys())
        _distances = list(Vts.values())
        concept_ids = []
        distances = []
        for c, d in zip(_concept_ids, _distances):
            concept_ids.append(c)
            distances.append(d)
        assert(len(concept_ids) == len(distances))
        
        # Contruct tuples with triples
        triples = []
        for v, N in Ets.items():
            if v in concept_ids:
                for u, rels in N.items():
                    if u in concept_ids:
                        triples.append((u, rels, v))
        
        target_ids = [self.concept2id[t_cpt] for t_cpt in target_concepts]   #id's of nodes in the concept graph of target concepts

        # Construct a list with labels; if the T-hop concept appears in target, then corresponding label is 1
        labels = []
        found_num = 0
        for c in concept_ids:  
            if c in target_ids:
                found_num += 1
                labels.append(1)
            else:
                labels.append(0)
        
        # Translate concept id's and relation id's back to text for interpretability
        concepts = [self.id2concept[c] for c in concept_ids]   # concept strings of concepts within T hops of source
        triples_text = [(self.id2concept[u], [self.id2relation[r] for r in rels], self.id2concept[v]) for (u, rels, v) in triples]

        logging.debug("\tReturn {} concepts and {} triples".format(len(concepts), len(triples)))
        return {"concepts":concepts, "labels":labels, "distances":distances, "triples":triples}, found_num, len(concepts)


    def filter_directed_triple(self, related_concepts, max_concepts=64, max_triples=256, max_neighbors=8):

        concepts = related_concepts['concepts']
        labels = related_concepts['labels']
        distances = related_concepts['distances']
        triples = related_concepts['triples']

        concept_ids = [self.concept2id[c] for c in concepts]

        triple_dict = {}
        for t in triples:
            head, tail = t[0], t[-1]
            head_index = concept_ids.index(head)
            tail_index = concept_ids.index(tail)
            if distances[head_index] <= distances[tail_index]:
                if t[-1] not in triple_dict:
                    triple_dict[t[-1]] = [t]
                else:
                    if len(triple_dict[t[-1]]) < max_neighbors:
                        triple_dict[t[-1]].append(t)

        starts = []
        for l, id in zip(labels, concept_ids):
            if l == 1:
                starts.append(id)

        sources = []
        for d, id in zip(distances, concept_ids):
            if d == 0:
                sources.append(id)

        shortest_paths = []
        for start in starts:
            shortest_paths.extend(bfs(start, triple_dict, sources))

        ground_truth_triples = []
        for path in shortest_paths:
            for i, n in enumerate(path[:-1]):
                ground_truth_triples.append((n, path[i+1]))
        ground_truth_triples_set = set(ground_truth_triples)

        _triples = []
        triple_labels = []
        for k,v in triple_dict.items():
            for t in v:
                _triples.append(t)
                if (t[-1], t[0]) in ground_truth_triples_set:
                    triple_labels.append(1)
                else:
                    triple_labels.append(0)

        concepts = concepts[:max_concepts]
        _triples = _triples[:max_triples]
        triple_labels = triple_labels[:max_triples]

        heads = []
        tails = []
        for triple in _triples:
            heads.append(concept_ids.index(triple[0]))
            tails.append(concept_ids.index(triple[-1]))

        related_concepts['relation_ids'] = [triple[1][0] for triple in _triples] # Keep only one relation
        related_concepts['head_idx'] = heads
        related_concepts['tail_idx'] = tails
        related_concepts['triple_labels'] = triple_labels
        related_concepts.pop('triples')
            
        return related_concepts


def bfs(start, triple_dict, source):
    """
        Perform breath-first-search
    """
    paths = [[[start]]]
    stop = False
    shortest_paths = []
    count = 0
    while 1:
        last_paths = paths[-1]
        new_paths = []
        for path in last_paths:
            if triple_dict.get(path[-1], False):
                triples = triple_dict[path[-1]]
                for triple in triples:
                    new_paths.append(path + [triple[0]])

        #print(new_paths)
        for path in new_paths:
            if path[-1] in source:
                stop = True
                shortest_paths.append(path)
        
        if count == 2:
            break
        paths.append(new_paths)
        count += 1
    
    return shortest_paths

