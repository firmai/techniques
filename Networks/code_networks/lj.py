# This proram performs CAD for a LiveJournal community
import urllib.request, os.path, pickle # Download and cache
import nltk # Convert text to terms
import networkx as nx, community # Build and analyze the network
import pandas as pd, numpy as np # Data science power tools

def download(base, domain_name):
    """
    Download interest data from the domain_name community on
    LiveJournal, convert interests to tags, create a domain DataFrame
    """
    members_url = "{}/fdata.bml?user={}&comm=1".format(base, domain_name) #(1)
    members = pd.read_table(members_url, sep=" ", 
                            comment="#", names=("direction", "uid"))

    wnl = nltk.WordNetLemmatizer() #(2)
    stop = set(nltk.corpus.stopwords.words('english')) | set(('&'))
    term_vectors = []#(3)
    for user in members.uid.unique():#(4)
        print("Loading {}".format(user)) # Progress indicator
        user_url = "{}/interestdata.bml?user={}".format(base, user)

        try: #(5)
            with urllib.request.urlopen(user_url) as source:
                raw_interests = [line.decode().lower().strip() 
                                 for line in source.readlines()]
        except:
            print("Could not open {}".format(user_url)) # Error message
            continue

        if raw_interests[0] == '! invalid user, or no interests':
            continue

        interests = [" ".join(wnl.lemmatize(w) #(6)
                              for w in nltk.wordpunct_tokenize(line)[2:] 
                              if w not in stop)
                     for line in raw_interests 
                     if line and line[0] != "#"]

        interests = set(interest for interest in interests if interest)#(7)
        term_vectors.append(pd.Series(index=interests, name=user).fillna(1))#(8)

    return pd.DataFrame().join(term_vectors, how="outer").fillna(0)\
        .astype(int)#(9)

LJ_BASE = "http://www.livejournal.com/misc"
DOMAIN_NAME = "thegoodwife_cbs"

cache_d = "cache/" + DOMAIN_NAME + ".pickle"
if not os.path.isfile(cache_d):
    domain = download(LJ_BASE, DOMAIN_NAME)
    if not path.os.isdir("cache"):
        os.mkdir("cache")
    with open(cache_d, "wb") as ofile:
        pickle.dump(domain, ofile)
else:
    with open(cache_d, "rb") as ifile:
        domain = pickle.load(ifile)

MIN_SUPPORT = 10
sums = domain.sum(axis=1)
limited = domain[sums >= MIN_SUPPORT]

cooc = limited.dot(limited.T) * (1 - np.eye(limited.shape[0]))

SLICING = 6
weights = cooc[cooc >= SLICING]
weights = weights.stack()
weights = weights / weights.max()
cd_network = weights.to_dict()
cd_network = {key:float(value) for key,value in cd_network.items()}

tag_network = nx.Graph()
tag_network.add_edges_from(cd_network)
nx.set_edge_attributes(tag_network, "weight", cd_network)

sizes = {key:float(value) for key,value in 
         domain.ix[tag_network].sum(axis=1).items()}
nx.set_node_attributes(tag_network, "w", sizes)

partition = community.best_partition(tag_network)
print("Modularity: {}".format(community.modularity(partition,
                              tag_network)))
nx.set_node_attributes(tag_network, "part", partition)

if not os.path.isdir("results"):
    os.mkdir("results")

with open("results/" + DOMAIN_NAME + ".graphml", "wb") as ofile:
    nx.write_graphml(tag_network, ofile)

HOW_MANY = 5
def describe_cluster(terms_df):
    # terms_df is a DataFrame; select the matching rows from "domain"
    rows = domain.join(terms_df, how="inner")
    # Calculate row sums, sort them, get the last HOW_MANY
    top_N = rows.sum(axis=1).sort_values(ascending=False)[:HOW_MANY]
    # What labels do they have?
    return top_N.index.values

tag_clusters = pd.DataFrame({"part_id" : pd.Series(partition)})
results = tag_clusters.groupby("part_id").apply(describe_cluster)
for r in results:
  print("-- {}".format("; ".join(r.tolist())))
