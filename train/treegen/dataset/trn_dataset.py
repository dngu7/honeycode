import glob
import logging
import os
import pickle
import time
import random
import numpy as np
from tqdm import tqdm
import networkx as nx
from collections import defaultdict
import torch.nn.functional as F
import torch

logger = logging.getLogger('gen')

class TrnDataset(object):
  def __init__(self, config, graphs, tag='train'):
    self.config = config
    self.data_path = config.dataset.data_path
    self.model_name = config.model.model_name
    self.max_num_nodes = config.dataset.dataset_max_num_nodes

    self.graphs = graphs
    self.num_graphs = len(graphs)
    self.npr = np.random.RandomState(config.seed)
    self.node_order = config.dataset.node_order
    self.num_canonical_order = config.model.num_canonical_order
    self.tag = tag
    self.num_fwd_pass = config.dataset.num_fwd_pass
    self.is_sample_subgraph = config.dataset.is_sample_subgraph
    self.num_subgraph_batch = config.dataset.num_subgraph_batch
    self.is_overwrite_precompute = config.dataset.is_overwrite_precompute
    self.is_tril = config.dataset.is_tril

    if self.is_sample_subgraph:
      assert self.num_subgraph_batch > 0
    
    self.save_path = os.path.join(
        self.data_path, '{}_{}_{}_precompute'.format(
            config.model.model_name, config.dataset.dataset_name, tag
            ))

    logger.debug("Entry at {}".format(__name__))

    if not os.path.isdir(self.save_path) or self.is_overwrite_precompute:
      self.file_names = []
      if not os.path.isdir(self.save_path):
        os.makedirs(self.save_path)

      logger.debug("Saving precomputed graphs at {}".format(self.save_path))
      self.config.dataset.save_path = self.save_path
      for index in tqdm(range(self.num_graphs)):
        G = self.graphs[index]
        data = self._get_graph_data(G)
        tmp_path = os.path.join(self.save_path, '{}_{}.p'.format(tag, index))
        pickle.dump(data, open(tmp_path, 'wb'))
        self.file_names += [tmp_path]
    else:
      self.file_names = glob.glob(os.path.join(self.save_path, '*.p'))

  def _get_graph_data(self, DG):
    
    assert any(n in [
      'topological',
      'lex_topological',
      'all'
    ] for n in self.node_order), "Invalid node ordering selected."

    adj_list = []
    if 'topological' in self.node_order:
      nodelist = [n for n in nx.topological_sort(DG)]
      
      adj = np.transpose(np.array(nx.to_numpy_matrix(DG, nodelist=nodelist)))
      if self.is_tril:
        adj = adj + adj.transpose()
      assert adj.sum() > 0, "{}".format(DG.graph)
      adj_list.append(adj)
    
    if 'lex_topological' in self.node_order:
      nodelist = [n for n in nx.lexicographical_topological_sort(DG)]

      adj = np.transpose(np.array(nx.to_numpy_matrix(DG, nodelist=nodelist)))
      if self.is_tril:
        adj = adj + adj.transpose()
      assert adj.sum() > 0, "{}".format(DG.graph)
      adj_list.append(adj)

    if 'all' in self.node_order:
      all_nodelist = list(nx.all_topological_sorts(DG))

      for nodelist in all_nodelist:
        adj = np.transpose(np.array(nx.to_numpy_matrix(DG, nodelist=nodelist)))
        if self.is_tril:
          adj = adj + adj.transpose()
        assert adj.sum() > 0, "{}".format(DG.graph)
        adj_list.append(adj)    

    return adj_list

  def __getitem__(self, index):
    K = 1
    N = self.max_num_nodes
    S = 1

    # load graph
    adj_list = pickle.load(open(self.file_names[index], 'rb'))
    num_nodes = adj_list[0].shape[0]
    num_subgraphs = int(np.floor((num_nodes - K) / S))

    max_val = 0

    if self.is_sample_subgraph:
      if self.num_subgraph_batch < num_subgraphs:
        num_subgraphs_pass = int(
            np.floor(self.num_subgraph_batch / self.num_fwd_pass))
      else:
        num_subgraphs_pass = int(np.floor(num_subgraphs / self.num_fwd_pass))

      end_idx = min(num_subgraphs, self.num_subgraph_batch)
    else:
      num_subgraphs_pass = int(np.floor(num_subgraphs / self.num_fwd_pass))
      end_idx = num_subgraphs

    ### random permute subgraph
    rand_perm_idx = self.npr.permutation(num_subgraphs).tolist()

    start_time = time.time()
    data_batch = []
    for ff in range(self.num_fwd_pass):
      ff_idx_start = num_subgraphs_pass * ff
      if ff == self.num_fwd_pass - 1:
        ff_idx_end = end_idx
      else:
        ff_idx_end = (ff + 1) * num_subgraphs_pass

      rand_idx = [random.choice(range(2, num_nodes - K))]
    
      #print("rand_idx", rand_idx)

      edges = []
      node_idx_gnn = []
      node_idx_feat = []
      label = []      
      subgraph_size = []
      subgraph_idx = []
      att_idx = []
      node_dist = []
      subgraph_count = 0


      for ii in range(len(adj_list)):
        # loop over different orderings
        adj_full = adj_list[ii]

        idx = -1
        for jj in range(1, num_nodes, S):
          # loop over different subgraphs
          idx += 1

          if jj + K > num_nodes:
            break

          if idx not in rand_idx:
            continue

          #Obtain distance from root
          adj_dist = np.tril(adj_full[:jj, :jj], k=-1).transpose()
          new_g = nx.from_numpy_matrix(np.asmatrix(adj_dist), create_using=nx.DiGraph())
          assert nx.is_arborescence(new_g)
          first = [nx.shortest_path_length(new_g, 0, n) + 1 for n in new_g.node]
          second = np.zeros(N - jj)
          node_dist += [np.concatenate([first, second])]


          adj_block = np.pad(
              adj_full[:jj, :jj], ((0, K), (0, 0)),
              'constant',
              constant_values=1.0
              )   # assuming fully connected for the new block

          adj_block = np.pad(
              adj_block, ((0, 0), (0, K)),
              'constant',
              constant_values=0.0)

          #Gather edges
          adj_block = torch.from_numpy(adj_block).to_sparse()

          edges += [adj_block.coalesce().indices().long()]

          if jj == 0:
            att_idx += [np.arange(1, K + 1).astype(np.uint8)]
          else:
            att_idx += [
                np.concatenate([
                    np.zeros(jj).astype(np.uint8),
                    np.arange(1, K + 1).astype(np.uint8)
                ])
            ]

          if jj == 0:
            node_idx_feat += [np.ones(K) * np.inf]
          else:
            node_idx_feat += [
                np.concatenate([np.arange(jj) + ii * N,
                                np.ones(K) * np.inf])
            ]
          
          
          idx_row_gnn, idx_col_gnn = np.meshgrid(
              np.arange(jj, jj + K), np.arange(jj + K))

          
          idx_row_gnn = idx_row_gnn.reshape(-1, 1)
          idx_col_gnn = idx_col_gnn.reshape(-1, 1)


          bot_gnn = np.concatenate([idx_row_gnn, idx_col_gnn],
                             axis=1).astype(np.int64)

          bot_gnn = np.stack([e for e in bot_gnn if e[0] != e[1]])

          node_idx_gnn += [bot_gnn]

          side_label = adj_full[idx_row_gnn, idx_col_gnn].flatten().astype(np.int64)[:-1]

          assert side_label.sum() == 1, "index:{} label error: {} \n {}".format(index, jj, adj_list)
          label += [side_label]

          subgraph_size += [jj + K]
          subgraph_idx += [
              np.ones_like(label[-1]).astype(np.int64) * subgraph_count
          ]
          subgraph_count += 1

      ### adjust index basis for the selected subgraphs
      cum_size = np.int_(np.cumsum([0] + subgraph_size))
      
      
      for ii in range(len(edges)):
   
        edges[ii] += torch.tensor(cum_size[ii])
        node_idx_gnn[ii] += cum_size[ii]

      data = {}
      data['adj'] = np.stack(adj_list, axis=0)

      data['edges'] = torch.cat(edges, dim=1).t().long()
      data['node_idx_gnn'] = np.concatenate(node_idx_gnn)
      data['node_idx_feat'] = np.concatenate(node_idx_feat)
      data['label'] = np.concatenate(label)
      data['att_idx'] = np.concatenate(att_idx)
      data['subgraph_idx'] = np.concatenate(subgraph_idx)
      data['subgraph_count'] = subgraph_count
      data['num_nodes'] = num_nodes
      data['subgraph_size'] = subgraph_size
      data['num_count'] = sum(subgraph_size)
      data['node_dist'] = np.concatenate(node_dist)
      data_batch += [data]


    end_time = time.time()

    return data_batch

  def __len__(self):
    return self.num_graphs

  def collate_fn(self, batch):
    assert isinstance(batch, list)
    start_time = time.time()
    batch_size = len(batch)
    N = self.max_num_nodes
    C = self.num_canonical_order
    batch_data = []

    for ff in range(self.num_fwd_pass):
      data = {}
      batch_pass = []
      for bb in batch:
        batch_pass += [bb[ff]]
      

      pad_size = [self.max_num_nodes - bb['num_nodes'] for bb in batch_pass]
      subgraph_idx_base = np.array([0] +
                                   [bb['subgraph_count'] for bb in batch_pass])
      subgraph_idx_base = np.cumsum(subgraph_idx_base)

      data['num_nodes_gt'] = torch.from_numpy(
          np.array([bb['num_nodes'] for bb in batch_pass])).long().view(-1)


      data['adj'] = torch.from_numpy(
          np.stack(
              [
                  np.pad(
                      bb['adj'], ((0, 0), (0, pad_size[ii]), (0, pad_size[ii])),
                      'constant',
                      constant_values=0.0) for ii, bb in enumerate(batch_pass)
              ],
              axis=0)).float()  # B X C X N X N


      data['node_dist'] = torch.from_numpy(
          np.stack(
              [
                  bb['node_dist'] for ii, bb in enumerate(batch_pass)
              ],
              axis=0)).long() 
      

      idx_base = np.array([0] + [bb['num_count'] for bb in batch_pass])
      idx_base = np.cumsum(idx_base)

      data['edges'] = torch.cat(
          [bb['edges'] + idx_base[ii] for ii, bb in enumerate(batch_pass)],
          dim=0)

      data['node_idx_gnn'] = torch.from_numpy(
          np.concatenate(
              [
                  bb['node_idx_gnn'] + idx_base[ii]
                  for ii, bb in enumerate(batch_pass)
              ],
              axis=0)).long()

      data['att_idx'] = torch.from_numpy(
          np.concatenate([bb['att_idx'] for bb in batch_pass], axis=0)).long()

      # shift one position for padding 0-th row feature in the model
      node_idx_feat = np.concatenate(
          [
              bb['node_idx_feat'] + ii * C * N
              for ii, bb in enumerate(batch_pass)
          ],
          axis=0) + 1


      node_idx_feat[np.isinf(node_idx_feat)] = 0
      node_idx_feat = node_idx_feat.astype(np.int64)

      data['node_idx_feat'] = torch.from_numpy(node_idx_feat).long()

      data['label'] = torch.from_numpy(
          np.concatenate([bb['label'] for bb in batch_pass])).float()


      data['subgraph_idx'] = torch.from_numpy(
          np.concatenate([
              bb['subgraph_idx'] + subgraph_idx_base[ii]
              for ii, bb in enumerate(batch_pass)
          ])).long()

      batch_data += [data]

    end_time = time.time()
    
    return batch_data