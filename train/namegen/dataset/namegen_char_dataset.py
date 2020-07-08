import logging
import sys
import os
import pickle
import time
import glob
import shutil
import numpy as np
import torch
from tqdm import tqdm
import unicodedata
import string


logger = logging.getLogger('namegen')


class NamegenCharDataset(object):
  def __init__(self, config, graphs, tag):
    self.config = config
    self.device = config.device
    self.cpucores = config.cpucores
    self.tag = tag
    self.dataset_name = config.dataset.dataset_name
    self.data_path = config.dataset.data_path
    
    self.all_letters = string.printable
    self.n_letters = len(self.all_letters) + 1 #EOS MARKER

    self.max_depth = config.dataset.max_depth
    self.max_child = config.dataset.max_child
    self.max_neigh = config.dataset.max_neigh
    self.node_types = config.dataset.node_types

    self.end_token = config.dataset.end_token
    
    self.model_name = config.model.model_name
    
    ## SETTINGS
    self.limit_precompute_inputs = config.dataset.limit_precompute_inputs

    self.batch_size = config.dataset.batch_size

    self.graphs = graphs
    self.num_graphs = len(graphs)
    #self.is_overwrite_precompute_repos = config.dataset.is_overwrite_precompute_repos
    self.is_overwrite_precompute_inputs = config.dataset.is_overwrite_precompute_inputs
    
    self.save_path = os.path.join(
        self.data_path, '{}_precompute'.format(
            self.dataset_name))
    
    #self.save_dir_one = os.path.join(self.save_path, '{}_repos'.format(tag))
    self.save_dir_two = os.path.join(self.save_path, '{}_inputs'.format(tag))
    
    for f in [self.save_path, self.save_dir_two]:
      if not os.path.exists(f):
        os.mkdir(f)
        logger.info("Created new dir: {}".format(f))
  
    if self.is_overwrite_precompute_inputs:
      shutil.rmtree(self.save_dir_two)
      os.mkdir(self.save_dir_two)
      self.start_precompute_inputs()

    self.precomputed_inputs = glob.glob(os.path.join(self.save_dir_two, '*.p'))
    logger.debug("Total file count: {}".format(len(self.precomputed_inputs)))


  def start_precompute_inputs(self):
    logger.info("Precomputing chars - singlethread mode")
    self._singlethread_precompute_inputs()
  
  def _singlethread_precompute_inputs(self):
    for graph_idx, graph in tqdm(enumerate(self.graphs), total=len(self.graphs)):
      self.precompute_inputs(graph, graph_idx)

  def precompute_inputs(self, graph, graph_idx):
    #start_time = time.time()

    edges = graph.edges
    #Compute the children count and parents
    children_dict = {}
    parent_dict = {}
    for e in edges:
      if e[0] not in children_dict:
        children_dict[e[0]] = []
      children_dict[e[0]].append(e[1])
      parent_dict[e[1]] = e[0]
    
    for node_idx, n in enumerate(graph.nodes.data()):
      node_id = n[0]
      node_type = n[1]['node_type'].decode("utf-8")
      path = n[1]['node_path'].decode("utf-8")
      path_split = path.split('/')
      #print(path_split)
      filename = path_split[-1]
      #parent_dir = path_split[1]
      #direct_parent_dir = path_split[-2]
      depth = min(self.max_depth, len(path_split) - 2) #minus parent and empty


      if depth == 0:
        #print(filename)
        filename = filename.split('-')[-1]

      #Convert to LONG
      name_ascii = [self.unicodeToAscii(c) for c in filename]
      name_idx = [self.all_letters.find(li) for li in name_ascii]

      label_idx = name_idx[1:]  + [self.n_letters - 1] #EOS
      #Children count
      if node_id in children_dict:
        child_count = min(self.max_child, len(children_dict[node_id]))
      else:
        child_count = 0

      #neighbor count
      if node_id in parent_dict:
        parent_id = parent_dict[node_id]
        neigh_count = min(self.max_neigh, len(children_dict[parent_id]))
      else:
        neigh_count = 0

      if len(name_idx) > 2:
        input_dict = {
          'input': name_idx, #List of integers
          'label': label_idx, 
          'node_type': [self.node_types.index(node_type)],
          'depth': [depth],
          'children_count': [child_count],
          'neighbor_count': [neigh_count]
        }

        input_file = '{}_{}-{}.p'.format(self.tag, graph_idx, node_idx)
        new_path = os.path.join(self.save_dir_two, input_file)
        pickle.dump(input_dict, open(new_path, 'wb'))

  def unicodeToAscii(self, s):
    '''From pytorch tutorial'''
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in self.all_letters
    )

  def __len__(self):
    return len(self.precomputed_inputs)

  def __getitem__(self, idx):

    input_file = self.precomputed_inputs[idx]
    with open(input_file, 'rb') as f:
      input_dict = pickle.load(f)

    input = torch.LongTensor(input_dict['input'])
    target = torch.LongTensor(input_dict['label'])
    node_type = torch.LongTensor(input_dict['node_type'])
    depth = torch.LongTensor(input_dict['depth'])
    n_child = torch.LongTensor(input_dict['children_count'])
    n_neigh = torch.LongTensor(input_dict['neighbor_count'])

    return (input, target, node_type, depth, n_child, n_neigh)


