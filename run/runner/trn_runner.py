import importlib.util
import logging
import os, sys
import time
import random
import pickle

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from utils.read_config import read_config
from utils.train_helper import load_model
from utils.vis_helper import draw_graph_list_separate

logger = logging.getLogger('gen')


class TrnRunner(object):
  def __init__(self, config, model_config_branch, model_name='gen'):

    logger.debug("{} initialized".format(__name__))

    self.model_name = model_name
    self.config = config
    self.use_gpu = config.use_gpu
    self.gpus = config.gpus
    self.device = config.device
    self.seed = config.seed
    self.random_gen = random.Random(self.seed)

    self.model_dir = os.path.join(self.config.models_dir, model_name)


    #config
    self.model_config_path = os.path.join(self.model_dir, 'config.yaml')
    assert os.path.exists(self.model_config_path), "Invalid config file: {}".format(self.model_config_path)    
    self.model_config = read_config(self.model_config_path)
    
    self.samples = config.samples
    self.batch_size = model_config_branch.batch_size
    #self.num_node_limit = model_config_branch.num_node_limit
    self.draw_settings = model_config_branch.draw_settings

    #snapshot
    self.model_snapshot = os.path.join(self.model_dir, model_config_branch.model_snapshot)
    assert os.path.exists(self.model_snapshot), "Invalid snapshot: {}".format(self.model_snapshot)
    
    #architecture
    self.model_file = model_config_branch.model_file
    self.model_arch = os.path.join(self.model_dir, self.model_file)
    assert os.path.exists(self.model_arch), "Invalid arch: {}".format(self.model_arch)

    #initialize module and model
    model_object = self.model_config.model.model_name
    spec = importlib.util.spec_from_file_location(
     model_object, self.model_arch
     )

    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    init_method = getattr(model_module, model_object)
    self.model_func = init_method(self.model_config)
    
    #load checkpoints
    load_model(self.model_func, self.model_snapshot, self.device)

    m_pkl = os.path.join(self.model_dir, 'metrics_gaussian_tv_0000100.pkl')
    self.metrics = pickle.load(open(m_pkl, "rb"))

  def get_directed_graph(self, adj):
    """Generates a directed graph based on adj"""

    horiz = np.all(adj == 0, axis=0)
    verti = np.all(adj == 0, axis=1)
    remove = np.logical_and(verti, horiz)
    adj = adj[~remove]
    adj = adj[:, ~remove]
    adj = np.asmatrix(adj)
    G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph())
    return G
    
  def eval(self):
    eval_time = time.time()
    self.model_func.to(self.device)
    logger.debug("Starting on device={}".format(self.model_func.device))
    self.model_func.eval() 

    num_test_batch = int(np.ceil(self.samples / self.batch_size))

    A_pred = []

    for _ in tqdm(range(num_test_batch)):
      with torch.no_grad():        
        start_time = time.time()
        input_dict = {}
        input_dict['is_sampling']=True
        input_dict['batch_size']=self.batch_size
        num_node_dist = np.array(self.metrics['train']['node_dist'])
        num_node_dist = np.bincount(num_node_dist)
        input_dict['num_nodes_pmf']= num_node_dist / num_node_dist.sum()

        A_tmp = self.model_func(input_dict)
        
        A_pred += [aa.data.cpu().numpy() for aa in A_tmp]
    
    #Convert to networkx
    vis_graphs = [self.get_directed_graph(aa) for aa in A_pred]
    
    #Rank graphs by number of nodes
    ranked_arbos = [(gg, gg.number_of_nodes()) for gg in vis_graphs]
    ranked_arbos = sorted(ranked_arbos, key=lambda x: [1], reverse=True)

    #Only keep x samples
    ranked_arbos = [gg[0] for gg in ranked_arbos][:self.samples]


    if self.draw_settings in ['all', 'one']:

      if self.draw_settings == 'all':
        drawn_arbos = ranked_arbos
      elif self.draw_settings == 'one':
        drawn_arbos = [self.random_gen.choice(ranked_arbos)]

      save_fname = os.path.join(self.config.config_save_dir, 'sampled_trees.png')

      draw_graph_list_separate(
        drawn_arbos, 
        fname=save_fname[:-4],
        is_single=True, 
        layout='kamada'
        )
    
    logger.debug("Generated {} Tree [{:2.2f} s]".format(self.samples, time.time() - eval_time))

    return ranked_arbos

