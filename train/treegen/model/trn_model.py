###############################################################################
#
# Some code is adapted from https://github.com/lrjconan/GRAN
#
###############################################################################

import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):

  def __init__(self,
               msg_dim,
               node_state_dim,
               edge_feat_dim,
               num_prop=1,
               num_layer=1,
               has_attention=True,
               att_hidden_dim=128,
               has_residual=False,
               has_graph_output=False,
               output_hidden_dim=128,
               graph_output_dim=None,
               node_dist_feat_dim=64):
    super(GNN, self).__init__()
    self.msg_dim = msg_dim
    self.node_state_dim = node_state_dim
    self.edge_feat_dim = edge_feat_dim
    self.num_prop = num_prop
    self.num_layer = num_layer
    self.has_attention = has_attention
    self.has_residual = has_residual
    self.att_hidden_dim = att_hidden_dim
    self.has_graph_output = has_graph_output
    self.output_hidden_dim = output_hidden_dim
    self.graph_output_dim = graph_output_dim
    self.node_dist_feat_dim = node_dist_feat_dim

    self.update_func = nn.ModuleList([
         nn.LSTMCell(input_size=self.msg_dim, hidden_size=self.node_state_dim)
         for _ in range(self.num_layer)
    ])

    self.msg_func = nn.ModuleList([
        nn.Sequential(
            *[
                nn.Linear(self.node_state_dim + self.edge_feat_dim + self.node_dist_feat_dim,
                          self.msg_dim),                
                nn.ReLU(),
                nn.Linear(self.msg_dim, self.msg_dim)
            ]) for _ in range(self.num_layer)
    ])

    if self.has_attention:
      self.att_head = nn.ModuleList([
          nn.Sequential(
              *[
                  nn.Linear(self.node_state_dim + self.edge_feat_dim + self.node_dist_feat_dim,
                            self.att_hidden_dim),
                  nn.ReLU(),
                  nn.Linear(self.att_hidden_dim, self.msg_dim),
                  nn.Sigmoid()
              ]) for _ in range(self.num_layer)
      ])

    if self.has_graph_output:
      self.graph_output_head_att = nn.Sequential(*[
          nn.Linear(self.node_state_dim, self.output_hidden_dim),
          nn.ReLU(),
          nn.Linear(self.output_hidden_dim, 1),
          nn.Sigmoid()
      ])

      self.graph_output_head = nn.Sequential(
          *[nn.Linear(self.node_state_dim, self.graph_output_dim)])

  def _prop(self, state, state_c, edge, edge_feat, layer_idx, dist_feat):
    ### compute message
    state_diff = state[edge[:, 0], :] - state[edge[:, 1], :]

    #print("_prop | state_diff", state_diff.shape)

    if self.edge_feat_dim > 0:
      edge_input = torch.cat([state_diff, edge_feat], dim=1)
    else:
      edge_input = state_diff
    
    if self.node_dist_feat_dim > 0:
      edge_input = torch.cat([edge_input, dist_feat], dim=1)


    msg = self.msg_func[layer_idx](edge_input)    

    ### attention on messages
    if self.has_attention:
      att_weight = self.att_head[layer_idx](edge_input)
      msg = msg * att_weight

    ### aggregate message by sum
    state_msg = torch.zeros(state.shape[0], msg.shape[1]).to(state.device)
    scatter_idx = edge[:, [1]].expand(-1, msg.shape[1])
    state_msg = state_msg.scatter_add(0, scatter_idx, msg)

    ### state update
    state, state_c = self.update_func[layer_idx](state_msg, (state, state_c))
    return state, state_c

  def forward(self, node_feat, node_feat_c, edge, edge_feat, dist_feat, graph_idx=None):
    """
      N.B.: merge a batch of graphs as a single graph

      node_feat: N X D, node feature
      edge: M X 2, edge indices
      edge_feat: M X D', edge feature
      graph_idx: N X 1, graph indices
    """

    state, state_c = node_feat, node_feat_c
    prev_state, prev_state_c = state, state_c

    for ii in range(self.num_layer):
      if ii > 0:
        state = F.relu(state)

      for jj in range(self.num_prop):
        state, state_c = self._prop(state, state_c, edge, edge_feat=edge_feat, layer_idx=ii, dist_feat=dist_feat)

    if self.has_residual:
      state = state + prev_state
      state_c = state_c + prev_state_c

    if self.has_graph_output:
      num_graph = graph_idx.max() + 1
      node_att_weight = self.graph_output_head_att(state)
      node_output = self.graph_output_head(state)

      # weighted average
      reduce_output = torch.zeros(num_graph,
                                  node_output.shape[1]).to(node_feat.device)
      reduce_output = reduce_output.scatter_add(0,
                                                graph_idx.unsqueeze(1).expand(
                                                    -1, node_output.shape[1]),
                                                node_output * node_att_weight)

      const = torch.zeros(num_graph).to(node_feat.device)
      const = const.scatter_add(
          0, graph_idx, torch.ones(node_output.shape[0]).to(node_feat.device))

      reduce_output = reduce_output / const.view(-1, 1)

      return reduce_output
    else:
      return state

class TrnModel(nn.Module):
  """ Tree Recurrent Network """

  def __init__(self, config):
    super(TrnModel, self).__init__()
    self.config = config
    self.device = config.device
    self.min_num_nodes = config.dataset.dataset_min_num_nodes
    self.max_num_nodes = config.dataset.dataset_max_num_nodes
    self.hidden_dim = config.model.hidden_dim
    self.is_sym = config.model.is_sym
    self.num_GNN_prop = config.model.num_GNN_prop
    self.num_GNN_layers = config.model.num_GNN_layers
    self.edge_weight = config.model.edge_weight if hasattr(
        config.model, 'edge_weight') else 1.0
    self.has_attention = config.model.has_attention
    self.num_canonical_order = config.model.num_canonical_order
    self.output_dim = 1
    self.att_edge_dim = config.model.att_edge_dim
    self.init_tree = config.test.init_tree
    self.sm = nn.Softmax(dim=1)
    self.node_dist_feat_dim = config.model.node_dist_feat_dim

    self.output_theta = nn.Sequential(
        nn.Linear(self.hidden_dim, self.output_dim))
        

    self.embedding_dim = config.model.embedding_dim
    self.decoder_input = nn.Sequential(
        nn.Linear(self.max_num_nodes, self.embedding_dim))
    self.decoder_input_c = nn.Sequential(
        nn.Linear(self.max_num_nodes, self.embedding_dim))


    self.decoder = GNN(
        msg_dim=self.hidden_dim,
        node_state_dim=self.hidden_dim,
        edge_feat_dim=2 * self.att_edge_dim,
        num_prop=self.num_GNN_prop,
        num_layer=self.num_GNN_layers,
        has_attention=self.has_attention,
        node_dist_feat_dim=2 * self.node_dist_feat_dim)

    ### Loss functions
    pos_weight = torch.ones([1]) * self.edge_weight

    self.adj_loss_func = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight, reduction='none')

  def _inference(self,
                 A_pad=None,
                 edges=None,
                 node_idx_gnn=None,
                 node_idx_feat=None,
                 att_idx=None,
                 node_dist=None):

    """ generate adj in row-wise auto-regressive fashion """
    B, C, N_max, _ = A_pad.shape

    A_pad = A_pad.view(B * C * N_max, -1)
    node_dist = node_dist.view(B * C * N_max, 1)

    node_feat = self.decoder_input(A_pad)  # BCN_max X H
    node_feat_c = self.decoder_input_c(A_pad)

    ### GNN inference
    # pad zero as node feature for newly generated nodes (1st row)
    node_feat = F.pad(
        node_feat, (0, 0, 1, 0), 'constant', value=0.0)  # (BCN_max + 1) X N_max

    node_feat_c = F.pad(
        node_feat_c, (0, 0, 1, 0), 'constant', value=0.0)  # (BCN_max + 1) X N_max

    # create symmetry-breaking edge feature for the newly generated nodes
    att_idx = att_idx.view(-1, 1)

    att_edge_feat = torch.zeros(edges.shape[0],
                                2 * self.att_edge_dim).to(node_feat.device)

    att_edge_feat = att_edge_feat.scatter(1, att_idx[[edges[:, 0]]], 1)
    att_edge_feat = att_edge_feat.scatter(
        1, att_idx[[edges[:, 1]]] + self.att_edge_dim, 1)

    if self.node_dist_feat_dim > 0:
      node_dist_feat = torch.zeros(edges.shape[0], 2 * self.node_dist_feat_dim).to(node_feat.device)
      x = node_dist[edges[:, 0]]
      y = node_dist[edges[:, 1]]


      node_dist_feat = node_dist_feat.scatter(1, x, 1)
      node_dist_feat = node_dist_feat.scatter(
        1, y + self.node_dist_feat_dim, 1)
    else:
      node_dist_feat = None
      

    
    node_state = self.decoder(
        node_feat[node_idx_feat], 
        node_feat_c[node_idx_feat], 
        edges, 
        edge_feat=att_edge_feat,
        dist_feat=node_dist_feat
        )

    diff = node_state[node_idx_gnn[:, 0], :] - node_state[node_idx_gnn[:, 1], :]

    log_theta = self.output_theta(diff)  # B X (tt+K)K

    log_theta = log_theta.view(-1, 1)  # B X CN(N-1)/2 X K

    return log_theta

  def _sampling(self, B):
    """ generate adj in row-wise auto-regressive fashion """

    K = 1
    S = 1
    H = self.hidden_dim
    N = self.max_num_nodes
    mod_val = (N - K) % S
    if mod_val > 0:
      N_pad = N - K - mod_val + int(np.ceil((K + mod_val) / S)) * S
    else:
      N_pad = N

    A = torch.zeros(B, N_pad, N_pad).to(self.device)
    dim_input = self.embedding_dim 

    #Initalize tree with single folder and one node
    if self.init_tree:
      A[:, :2, :2] = torch.tensor([[0.,1.], [1.,0.]])
      start_loop = 2
    else:
      start_loop = 0

    ### cache node state for speed up
    node_state = torch.zeros(B, N_pad, dim_input).to(self.device)
    node_state_c = torch.zeros(B, N_pad, dim_input).to(self.device)

    for ii in range(start_loop, N_pad, S):
      jj = ii + K
      if jj > N_pad:
        break
      A[:, ii:, :] = .0
      A[:, :, ii:] = .0
      
      #Create balanced matrix
      dia_A = torch.tril(A, diagonal=-1)
      A = dia_A + torch.transpose(dia_A, 1, 2)

      #Calculate depth
      if self.node_dist_feat_dim > 0:
        np_adj = torch.transpose(dia_A[:, :ii, :ii], 1, 2).cpu().numpy()
        node_dist = []
        for bb in range(B):
          new_g = nx.from_numpy_matrix(np.asmatrix(np_adj[bb]), create_using=nx.DiGraph())
          first = torch.tensor(
                  [nx.shortest_path_length(new_g, 0, n) + 1
                    for n in new_g.node]).long()
          second = torch.zeros(K).long()
          node_dist.append(
            torch.cat([first, second],dim=0))
        node_dist = torch.cat(node_dist, dim=0).view(-1, 1).to(A.device)

      if ii <= start_loop:
        node_state[:, ii - K:ii, :] = self.decoder_input(A[:, ii - K:ii, :])
        node_state_c[:, ii - K:ii, :] = self.decoder_input_c(A[:, ii - K:ii, :])
          
      else:
        node_state[:, :ii, :] = self.decoder_input(A[:, :ii, :N])
        node_state_c[:, :ii, :] = self.decoder_input_c(A[:, :ii, :N])

      #Create extra row of zeros
      node_state_in = F.pad(
          node_state[:, :ii, :], (0, 0, 0, K), 'constant', value=.0)
      node_state_in_c = F.pad(
          node_state_c[:, :ii, :], (0, 0, 0, K), 'constant', value=.0)

      ### propagation
      adj = F.pad(
          A[:, :ii, :ii], (0, K, 0, K), 'constant', value=1.0)  # B X jj X jj

      adj = torch.tril(adj, diagonal=-1)
      adj = adj + torch.transpose(adj, 1,2)

      edges = [
          adj[bb].to_sparse().coalesce().indices() + bb * adj.shape[1]
          for bb in range(B)
      ]
      edges = torch.cat(edges, dim=1).t()

      att_idx = torch.cat([torch.zeros(ii).long(),
                           torch.arange(1, K + 1)]).to(self.device)
      att_idx = att_idx.view(1, -1).expand(B, -1).contiguous().view(-1, 1)

      # create one-hot feature
      att_edge_feat = torch.zeros(edges.shape[0],
                                  2 * self.att_edge_dim).to(self.device)
      att_edge_feat = att_edge_feat.scatter(1, att_idx[[edges[:, 0]]], 1)
      att_edge_feat = att_edge_feat.scatter(
          1, att_idx[[edges[:, 1]]] + self.att_edge_dim, 1)


      if self.node_dist_feat_dim > 0:
        node_dist_feat = torch.zeros(edges.shape[0], 2 * self.node_dist_feat_dim).to(node_dist.device)
        x = node_dist[edges[:, 0]]
        y = node_dist[edges[:, 1]]

        x = torch.clamp(x, 0, self.node_dist_feat_dim - 1) 
        y = torch.clamp(y, 0, self.node_dist_feat_dim - 1) 
        node_dist_feat = node_dist_feat.scatter(1, x, 1)
        node_dist_feat = node_dist_feat.scatter(
          1, y + self.node_dist_feat_dim, 1)
      else:
        node_dist_feat = None


      node_state_out = self.decoder(
          node_state_in.view(-1, H), 
          node_state_in_c.view(-1,H), 
          edges, edge_feat=att_edge_feat,
          dist_feat=node_dist_feat
          )
      node_state_out = node_state_out.view(B, jj, -1)
      
      idx_row, idx_col = np.meshgrid(np.arange(ii, jj), np.arange(jj))
      idx_row = torch.from_numpy(idx_row.reshape(-1)[:-1]).long().to(self.device)
      idx_col = torch.from_numpy(idx_col.reshape(-1)[:-1]).long().to(self.device)
      
      diff = node_state_out[:,idx_row, :] - node_state_out[:,idx_col, :]


      diff = diff.view(-1, node_state.shape[2])
      log_theta = self.output_theta(diff)

      log_theta = log_theta.view(B, -1)

      prob = self.sm(log_theta)

      new_matrix_border = torch.multinomial(prob, 1)

      predictions = new_matrix_border.squeeze().long()

      one_hot = torch.zeros(B, ii)
      one_hot[torch.arange(B), predictions] = 1
      one_hot = one_hot.view(B, 1, ii)

      A[:, ii:jj, :ii] = one_hot

    dia_A = torch.tril(A, diagonal=-1)
    A = torch.transpose(dia_A, 1, 2)
    return A

  def forward(self, input_dict):
    """
      B: batch size
      N: number of rows/columns in mini-batch
      N_max: number of max number of rows/columns
      M: number of augmented edges in mini-batch
      H: input dimension of GNN 
      K: block size
      E: number of edges in mini-batch
      S: stride
      C: number of canonical orderings

      Args:
        A_pad: B X C X N_max X N_max, padded adjacency matrix         
        node_idx_gnn: M X 2, node indices of augmented edges
        node_idx_feat: N X 1, node indices of subgraphs for indexing from feature
                      (0 indicates indexing from 0-th row of feature which is 
                        always zero and corresponds to newly generated nodes)  
        att_idx: N X 1, one-hot encoding of newly generated nodes
                      (0 indicates existing nodes, 1-D indicates new nodes in
                        the to-be-generated block)
        subgraph_idx: E X 1, indices corresponding to augmented edges
                      (representing which subgraph in mini-batch the augmented 
                      edge belongs to)
        edges: E X 2, edge as [incoming node index, outgoing node index]
        label: E X 1, binary label of augmented edges        
        num_nodes_pmf: N_max, empirical probability mass function of number of nodes

      Returns:
        loss                        if training
        list of adjacency matrices  else
    """
    is_sampling = input_dict[
        'is_sampling'] if 'is_sampling' in input_dict else False
    batch_size = input_dict[
        'batch_size'] if 'batch_size' in input_dict else None
    A_pad = input_dict['adj'] if 'adj' in input_dict else None
    node_idx_gnn = input_dict[
        'node_idx_gnn'] if 'node_idx_gnn' in input_dict else None
    node_idx_feat = input_dict[
        'node_idx_feat'] if 'node_idx_feat' in input_dict else None
    att_idx = input_dict['att_idx'] if 'att_idx' in input_dict else None    
    subgraph_idx = input_dict[
        'subgraph_idx'] if 'subgraph_idx' in input_dict else None
    edges = input_dict['edges'].long() if 'edges' in input_dict else None
    label = input_dict['label'] if 'label' in input_dict else None
    num_nodes_pmf = input_dict['num_nodes_pmf'] if 'num_nodes_pmf' in input_dict else None
    node_dist = input_dict['node_dist'] if 'node_dist' in input_dict else None

    N_max = self.max_num_nodes

    if not is_sampling:
      B, _, N, _ = A_pad.shape

      ### compute adj loss
      log_theta = self._inference(
          A_pad=A_pad,
          edges=edges,
          node_idx_gnn=node_idx_gnn,
          node_idx_feat=node_idx_feat,
          att_idx=att_idx,
          node_dist=node_dist)

      loss = self.adj_loss_func(log_theta, label.view(-1,1))
      return loss

    else:
      A = self._sampling(batch_size)

      ## sample number of nodes
      num_nodes_pmf = torch.from_numpy(num_nodes_pmf).to(self.device)
      num_nodes = torch.multinomial(
          num_nodes_pmf, batch_size, replacement=True) + 1  # shape B X 1

      A_list = [
          A[ii, :num_nodes[ii], :num_nodes[ii]] for ii in range(batch_size)
      ]
      return A_list
