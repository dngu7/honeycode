import os
import pickle

# save a list of graphs
def save_graph_list(G_list, fname):
  with open(fname, "wb") as f:
    pickle.dump(G_list, f)
    