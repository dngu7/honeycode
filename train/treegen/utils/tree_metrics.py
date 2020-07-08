import numpy as np
import networkx as nx


def tree_root(gg):
  root_node = None
  for n in gg:
    if len(gg.in_edges(n)) == 0:
      root_node = n
      break
  assert root_node != None, "Not a tree"
  return root_node

def nearest_descendants(gg, n):
  '''Returns list of  nodes closest to node'''
  return [e[1] for e in gg.out_edges(n)]

def depth_by_node(gg, root):
  '''Returns depth (key) and list of nodes (value) in dictionary'''
  node_depth = {}
  depth = 0

  #Find nodes for each depth
  stack = [(root, depth)]
  while stack:
    node, depth = stack.pop(0)

    if depth not in node_depth:
      node_depth[depth] = []
    node_depth[depth].append(node)

    #Gets all nodes in next depth connected to existing node
    desc = nearest_descendants(gg, node)
    # Add these to stack
    for d in desc:
      stack.append((d, depth + 1))
  
  return node_depth
  
def _depth_density(gg, root):
  '''Returns histogram (x-axis=depth, y-axis=count)'''
  node_depth = {}
  depth = 0

  #Find nodes for each depth
  stack = [(root, depth)]
  while stack:
    node, depth = stack.pop(0)

    if depth not in node_depth:
      node_depth[depth] = 0
    node_depth[depth] += 1

    #Gets all nodes in next depth connected to existing node
    desc = nearest_descendants(gg, node)
    # Add these to stack
    for d in desc:
      stack.append((d, depth + 1))
  
  return node_depth

def to_hist(dicts):
  assert isinstance(dicts, dict)
  hist = np.zeros(max(dicts.keys()) + 1)

  for d in dicts.keys():
    hist[d] = dicts[d]
  return hist

def depth_density(graphs, pmf=True):
  '''Returns the density of each depth level in histogram'''

  all_depth_density = []
  for gg in graphs:

    #Find root of tree
    root = tree_root(gg)

    #Organize nodes into depths using dictionary
    node_depth = _depth_density(gg, root)
    dist = to_hist(node_depth)
    if pmf:
      dist = dist / dist.sum()
    
    all_depth_density.append(dist)
  
  return all_depth_density

  

def node_dist(graphs, pmf=True):
  dist = np.bincount([len(gg.nodes) for gg in graphs])

  if pmf:
    dist = dist / np.sum(dist)
  
  return dist

def depth_dist(graphs, pmf=True):
  dist = np.bincount([len(nx.dag_longest_path(gg)) for gg in graphs])

  if pmf:
    dist = dist / np.sum(dist)
  
  return dist

def degree_dist(graphs, metric='mmd'):

  dist = [np.array(nx.degree_histogram(gg)) for gg in graphs]

  if metric == 'mmd':
    dist = [h / np.sum(h) for h in dist]
  elif metric == 'emd':
    dist = [np.mean(dist)]
  
  return dist
