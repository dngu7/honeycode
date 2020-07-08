import copy
import logging
import os
import pickle
import sys
import time
from collections import defaultdict

import networkx as nx
import numpy as np
import torch
import torch.optim as optim
from scipy import stats
from tqdm import tqdm

from dataset.trn_dataset import *
from model.trn_model import *
from preprocess import graph_loader
from utils import data_parallel, helpers
from utils.dist_helper import compute_mmd, gaussian_tv
from utils.train_helper import load_model, snapshot
from utils.tree_metrics import (degree_dist, depth_density, depth_dist,
                                node_dist)
from utils.vis_helper import draw_graph_list, draw_graph_list_separate

logger = logging.getLogger('gen')

class TrnRunner(object):

  def __init__(self, config):

    self.config = config
    self.seed = config.seed
    self.dataset_conf = config.dataset
    self.model_conf = config.model
    self.train_conf = config.train
    self.test_conf = config.test
    self.use_gpu = config.use_gpu
    self.gpus = config.gpus
    self.device = config.device
    self.is_vis = config.test.is_vis
    self.num_vis = config.test.num_vis
    self.vis_num_row = config.test.vis_num_row
    self.is_single_plot = config.test.is_single_plot
    self.num_gpus = len(self.gpus)
    self.is_shuffle = False

    if self.train_conf.is_resume:
      self.config.save_dir = self.train_conf.resume_dir

    self.graphs = graph_loader.load_graph(
      source=config.dataset.source,
      data_dir=config.dataset.data_path,
      min_num_nodes=config.dataset.dataset_min_num_nodes,
      max_num_nodes=config.dataset.dataset_max_num_nodes,
      node_labels=True,
      graph_labels=True
      )

    self.num_graphs = len(self.graphs)

    self.train_ratio = config.dataset.train_ratio
    self.dev_ratio = config.dataset.dev_ratio
    self.num_train = int(float(self.num_graphs) * self.train_ratio)
    self.num_dev = int(float(self.num_graphs) * self.dev_ratio)
    self.num_test_gt = self.num_graphs - self.num_train
    self.num_test_gen = config.test.num_test_gen

    logger.info('Train/val/test = {}/{}/{}'.format(
      self.num_train, 
      self.num_dev,
      self.num_test_gt
      ))

    ### shuffle all graphs
    if self.is_shuffle:
      self.npr = np.random.RandomState(self.seed)
      self.npr.shuffle(self.graphs)

    self.graphs_train = self.graphs[:self.num_train]
    self.graphs_dev = self.graphs[:self.num_dev]
    self.graphs_test = self.graphs[self.num_train:]

    #Save pre-calculated metrics for later
    self.pmf=True
    self.mmd_type='mmd'
    self.metrics = {'train': {}, 'test': {}, 'pred': {}, 'results': {}}

    self.metrics['train']['node_dist'] = [len(gg.nodes) for gg in self.graphs_train]
    self.metrics['train']['depth_dist'] = [len(nx.dag_longest_path(gg)) for gg in self.graphs_train]
    self.metrics['train']['degree_dist'] = degree_dist(self.graphs_train, metric=self.mmd_type)
    self.metrics['train']['depth_density'] = depth_density(self.graphs_train, pmf= self.pmf)

    self.metrics['test']['node_dist'] = [len(gg.nodes) for gg in self.graphs_test]
    self.metrics['test']['depth_dist'] = [len(nx.dag_longest_path(gg)) for gg in  self.graphs_test]
    self.metrics['test']['degree_dist'] = degree_dist(self.graphs_test, metric=self.mmd_type)
    self.metrics['test']['depth_density'] = depth_density(self.graphs_test, pmf= self.pmf)

    
    ### save split locally for benchmarking
    if config.dataset.is_save_split:      
      base_path = os.path.join(config.dataset.data_path, 'save_split')
      if not os.path.exists(base_path):
        os.makedirs(base_path)
      
      helpers.save_graph_list(
          self.graphs_train,
          os.path.join(base_path, '{}_train.p'.format(config.dataset.dataset_name)))
      helpers.save_graph_list(
          self.graphs_dev,
          os.path.join(base_path, '{}_dev.p'.format(config.dataset.dataset_name)))
      helpers.save_graph_list(
          self.graphs_test,
          os.path.join(base_path, '{}_test.p'.format(config.dataset.dataset_name)))

  def train(self):

    train_dataset = eval(self.dataset_conf.loader_name)(self.config, self.graphs_train, tag='train')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=self.train_conf.batch_size,
        shuffle=self.train_conf.shuffle,
        num_workers=self.train_conf.num_workers,
        collate_fn=train_dataset.collate_fn,
        drop_last=False
        )
    
    model = eval(self.model_conf.model_name)(self.config)
    
    #Parallelize gpu
    if self.use_gpu:
      model = data_parallel.DataParallel(model, device_ids=self.gpus).to(self.device)
    
    params = filter(lambda p: p.requires_grad, model.parameters())

    #Setup optimizer
    if self.train_conf.optimizer == 'SGD':
      optimizer = optim.SGD(
          params,
          lr=self.train_conf.lr,
          momentum=self.train_conf.momentum,
          weight_decay=self.train_conf.wd
          )
    elif self.train_conf.optimizer == 'Adam':
      optimizer = optim.Adam(
        params, 
        lr=self.train_conf.lr, 
        weight_decay=self.train_conf.wd
        )
    else:
      raise ValueError("Non-supported optimizer!")


    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=self.train_conf.lr_decay_epoch,
        gamma=self.train_conf.lr_decay)

    # reset gradient
    optimizer.zero_grad()

    # resume training
    resume_epoch = 0
    if self.train_conf.is_resume:
      model_file = os.path.join(self.train_conf.resume_dir,
                                self.train_conf.resume_model)
      load_model(
          model.module if self.use_gpu else model,
          model_file,
          self.device,
          optimizer=optimizer,
          scheduler=lr_scheduler)
      resume_epoch = self.train_conf.resume_epoch


    # Training Loop
    iter_count = 0    
    results = defaultdict(list)
    for epoch in range(resume_epoch, self.train_conf.max_epoch):
      model.train()
      
      train_iterator = train_loader.__iter__()

      for inner_iter in range(len(train_loader) // self.num_gpus):
        optimizer.zero_grad()

        batch_data = []
        if self.use_gpu:
          for _ in self.gpus:
            data = train_iterator.next()
            batch_data.append(data)
            iter_count += 1
        
        
        avg_train_loss = .0        
        for ff in range(self.dataset_conf.num_fwd_pass):
          batch_fwd = []
          
          if self.use_gpu:
            for dd, gpu_id in enumerate(self.gpus):
              data = {}
              data['adj'] = batch_data[dd][ff]['adj'].pin_memory().to(gpu_id, non_blocking=True)          
              data['edges'] = batch_data[dd][ff]['edges'].pin_memory().to(gpu_id, non_blocking=True)
              data['node_idx_gnn'] = batch_data[dd][ff]['node_idx_gnn'].pin_memory().to(gpu_id, non_blocking=True)
              data['node_idx_feat'] = batch_data[dd][ff]['node_idx_feat'].pin_memory().to(gpu_id, non_blocking=True)
              data['label'] = batch_data[dd][ff]['label'].pin_memory().to(gpu_id, non_blocking=True)
              data['att_idx'] = batch_data[dd][ff]['att_idx'].pin_memory().to(gpu_id, non_blocking=True)
              data['subgraph_idx'] = batch_data[dd][ff]['subgraph_idx'].pin_memory().to(gpu_id, non_blocking=True)
              data['node_dist'] = batch_data[dd][ff]['node_dist'].pin_memory().to(gpu_id, non_blocking=True)
              batch_fwd.append((data,))

          if batch_fwd:
            train_loss = model(*batch_fwd).mean()   
            avg_train_loss += train_loss              

            # assign gradient
            train_loss.backward()
        
        optimizer.step()
        avg_train_loss /= float(self.dataset_conf.num_fwd_pass)
        lr_scheduler.step()

        train_loss = float(avg_train_loss.data.cpu().numpy())
        
        results['train_loss'] += [train_loss]
        results['train_step'] += [iter_count]

        if iter_count % self.train_conf.display_iter == 0 or iter_count == 1:
          logger.info("Loss @ epoch {:04d} iteration {:08d} = {}".format(epoch + 1, iter_count, train_loss))
          
      if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
        snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1, scheduler=lr_scheduler)

    pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
    self.writer.close()
    
    return 1

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


  def compute_stats(self, group1, group2, kernel):
    logger.info("Calculating MMD : {} v {}".format(group1, group2))

    results = {}
    for _metric in self.metrics[group1].keys():
      
      if _metric in ['node_dist', 'depth_dist']:


        _dist_1 = np.bincount(self.metrics[group1][_metric])
        _dist_2 = np.bincount(self.metrics[group2][_metric])

        emd_val = gaussian_tv(
            _dist_1/_dist_1.sum(),
            _dist_2/_dist_2.sum()
        )
        
        results[_metric + '_emd'] = emd_val

      elif _metric in ['degree_dist', 'depth_density']:

        mmd = compute_mmd(
          samples1=self.metrics[group1][_metric],
          samples2=self.metrics[group2][_metric],
          kernel=kernel,
          is_hist=False,
          is_parallel=False
          )

        results[_metric + '_mmd'] = mmd
  
    return results



  def eval(self):
    test_epoch = self.test_conf.test_model_name
    test_epoch = test_epoch[test_epoch.rfind('_') + 1:test_epoch.find('.pth')]

    self.config.save_dir = self.test_conf.test_model_dir

    model = eval(self.model_conf.model_name)(self.config)
    model_file = os.path.join(self.config.save_dir, self.test_conf.test_model_name)

    logger.debug("Loading Model: {}".format(model_file))

    load_model(model, model_file, self.device)

    if self.use_gpu:
      model = torch.nn.DataParallel(model, device_ids=self.gpus).to(self.device)

    model.eval()

    ### Generate Graphs
    A_pred = []
    num_nodes_pred = []
    num_test_batch = int(np.ceil(self.num_test_gen / self.test_conf.batch_size))

    gen_run_time = []
    for ii in tqdm(range(num_test_batch)):
      with torch.no_grad():        
        start_time = time.time()
        input_dict = {}
        input_dict['is_sampling']=True
        input_dict['batch_size']=self.test_conf.batch_size
        num_node_dist = np.array(self.metrics['train']['node_dist'])
        num_node_dist = np.bincount(num_node_dist)
        input_dict['num_nodes_pmf']= num_node_dist / num_node_dist.sum()
        A_tmp = model(input_dict)
        gen_run_time += [time.time() - start_time]
        A_pred += [aa.data.cpu().numpy() for aa in A_tmp]
        num_nodes_pred += [aa.shape[0] for aa in A_tmp]

    avg_time = np.mean(gen_run_time)
    logger.info('Average test time per mini-batch = {}'.format(avg_time))

    ################################################
    ### Generate Graphs
    ################################################

    final_samples = [self.get_directed_graph(aa) for aa in A_pred]
    num_nodes_gen = [len(aa) for aa in final_samples]

    #########################
    ## METRICS
    #########################
    arb_count = sum([1 for g in final_samples if nx.is_arborescence(g)])
    logger.info("Total Samples: {} Tree: {}, %: {}".format(
      len(final_samples), arb_count, arb_count / len(final_samples)))

    #Calculate metrics for samples
    self.metrics['pred']['node_dist'] = [len(gg.nodes) for gg in final_samples]
    self.metrics['pred']['depth_dist'] = [len(nx.dag_longest_path(gg)) for gg in final_samples]
    self.metrics['pred']['degree_dist'] = degree_dist(final_samples, metric=self.mmd_type)
    self.metrics['pred']['depth_density'] = depth_density(final_samples, pmf= self.pmf)
    self.metrics['pred']['tree_ratio'] = arb_count / len(final_samples)

    self.metrics['train_stats'] = self.compute_stats('train', 'pred', gaussian_tv)
    logger.info("Train Stats: {}".format(self.metrics['train_stats']))
    self.metrics['test_stats'] = self.compute_stats('test', 'pred', gaussian_tv)
    logger.info("Test Stats: {}".format(self.metrics['test_stats']))

    self.metrics['pred_samples'] = final_samples

    save_name = os.path.join(self.config.save_dir, 'metrics_{}_{}.pkl'.format('gaussian_tv', test_epoch))
    pickle.dump(self.metrics, open(save_name, "wb"))

    ################################################
    ### Visualize Generated Graphs
    ################################################

    if self.is_vis:
      num_col = self.vis_num_row
      num_row = int(np.ceil(self.num_vis / num_col))
      test_epoch = self.test_conf.test_model_name
      test_epoch = test_epoch[test_epoch.rfind('_') + 1:test_epoch.find('.pth')]
      save_name = os.path.join(self.config.save_dir, '{}_gen_graphs_epoch_{}.png'.format(self.config.test.test_model_name[:-4], test_epoch))

      vis_graphs = final_samples

      if self.is_single_plot:
        draw_graph_list(
          vis_graphs[:self.num_vis], 
          num_row, 
          num_col, 
          fname=save_name, 
          layout='kamada')
      else:
        draw_graph_list_separate(
          vis_graphs[:self.num_vis], 
          fname=save_name[:-4],
          is_single=True, 
          layout='kamada')

      save_name = os.path.join(self.config.save_dir, 'train_graphs.png')

      if self.is_single_plot:
        draw_graph_list(
            self.graphs_train[:self.num_vis],
            num_row,
            num_col,
            fname=save_name,
            layout='kamada')
      else:      
        draw_graph_list_separate(
            self.graphs_train[:self.num_vis],
            fname=save_name[:-4],
            is_single=True,
            layout='kamada')
