import networkx as nx
import os
import numpy as np
from tqdm import tqdm
import copy

def ext(name):
  return {
    'julia': '.jl'
  }[name]

def load_graph(data_dir, min_num_nodes, max_num_nodes, node_labels, graph_labels):
  
  #Each file should contain the datasetname at the front of the file
  name = data_dir.split('/')[-1]

  #(node_x, node_y)
  data_adj = np.loadtxt(
    fname=os.path.join(data_dir, '{}_A.txt'.format(name)),
    delimiter='|').astype(int)

  
  if node_labels:
    #(node_id, **info)
    data_node_label = np.loadtxt(
      fname=os.path.join(data_dir, '{}_node_labels.txt'.format(name)),
      delimiter='|',
      dtype={
        'names': ('node_id', 'tree_id', 'node_type', 'node_name', 'node_path'),
        'formats': ('i4', 'i4', 'S4', 'S100', 'S250')
        }
      )

  else:
    #(node_id, graph_id)
    data_node_label = np.loadtxt(
      fname=os.path.join(data_dir, '{}_graph_indicators.txt'.format(name)),
      delimiter='|').astype(int)

  #(graph_id, **info)
  if graph_labels:
    data_graph_label = np.loadtxt(
      fname=os.path.join(data_dir, '{}_graph_labels.txt'.format(name)),
      delimiter='|',
      dtype={
        'names': ('tree_id', 'tree_name', 'language', 'stars', 'git_uri', 'last_update'),
        'formats': ('i4', 'S100', 'S100', 'i4', 'S250', 'S100')
        }
      ) 
  else:
    #(graph_id)
    data_node_label = np.loadtxt(
      fname=os.path.join(data_dir, '{}_graph_labels.txt'.format(name)),
      delimiter=',',
      usecols=(0)).astype(int)

  DG = nx.DiGraph()

  # Add Edges
  data_tuple = list(map(tuple, data_adj))
  DG.add_edges_from(data_tuple)

  # Add Nodes
  node_bar = tqdm(range(data_node_label.shape[0]))

  for i in node_bar:
    #node_bar.set_description("Processing node {}".format(i))

    if node_labels:
      DG.add_node(data_node_label[i][0],
        label=data_node_label[i][0],
        tree_id=data_node_label[i][1],
        node_type=data_node_label[i][2],
        node_name=data_node_label[i][3],
        node_path=data_node_label[i][4],
        )
    else:
      DG.add_node(data_node_label[i][0],
        label=data_node_label[i][0],
        tree_id=data_node_label[i][1]
      )

  isolates = list(nx.isolates(DG))
  selfloops = list(nx.selfloop_edges(DG))
  if len(isolates) or len(selfloops):
    print("Removing isolates ({}) and selfloops ({})".format(
      len(isolates),
      len(selfloops)
    ))
    DG.remove_nodes_from(isolates)
    DG.remove_edges_from(selfloops)

  tree_id_node_list = dict()
  tree_id_lang = dict()
  for n in DG.nodes.data():
    tree_id = n[1]['tree_id']

    if tree_id not in tree_id_node_list:
      tree_id_node_list[tree_id] = []
      tree_id_lang[tree_id] = False
    
    tree_id_node_list[tree_id].append(n[0])
    #check if .jl extension exists
    if ext(name) in n[1]['node_name'].decode("utf-8"):
      tree_id_lang[tree_id] = True

  graphs = []
  graph_bar = tqdm(range(data_graph_label.shape[0]))
  for i in graph_bar:
    #graph_bar.set_description("Processing graph {}".format(i))
    
    tree_id = data_graph_label[i][0]
    #Search for nodes with same tree-id
    nodes = tree_id_node_list[tree_id]
    #Language file exist
    lang = tree_id_lang[tree_id]

    #Create sub-graph
    G_sub = DG.subgraph(nodes).copy()
    G_sub.graph['label'] = tree_id
    
    #lang node reduces the number of additional steps
    if graph_labels:
      G_sub.graph['tree_id'] = tree_id
      G_sub.graph['tree_name'] = data_graph_label[i][1]
      G_sub.graph['language'] = data_graph_label[i][2]
      G_sub.graph['stars'] = data_graph_label[i][3]
      G_sub.graph['git_uri'] = data_graph_label[i][4]
      G_sub.graph['last_update'] = data_graph_label[i][5]
  
    if G_sub.number_of_nodes() >= min_num_nodes \
      and G_sub.number_of_nodes() <= max_num_nodes \
      and lang and nx.is_arborescence(G_sub):
      graphs.append(G_sub)
      

      #print(G_sub.graph['tree_name'], G_sub.graph['tree_id'])
  
  return graphs





 

  