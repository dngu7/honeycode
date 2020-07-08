import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUCharNameGenModel(nn.Module):
  def __init__(self,config, n_letters):
    super(GRUCharNameGenModel, self).__init__()
    self.max_depth = config.dataset.max_depth + 1
    self.max_child = config.dataset.max_child + 1
    self.max_neigh = config.dataset.max_neigh + 1
    self.node_types = config.dataset.node_types
    self.n_letters = n_letters
    self.input_size = n_letters

    self.nhid = config.model.nhid
    self.input_emb_size = config.model.input_emb_size
    self.depth_emb_size = config.model.depth_emb_size
    self.nodetype_emb_size = config.model.nodetype_emb_size

    self.num_layers = config.model.num_layers
    self.dropout = config.model.dropout
    self.bidirect = config.model.bidirectional
    self.hid_init = config.model.hid_init

    #Embedding Layers
    self.input_embed_layer = nn.Embedding(
      num_embeddings=self.n_letters,
      embedding_dim=self.input_emb_size
    ) 

    self.depth_embed_layer = nn.Embedding(
      num_embeddings=self.max_depth,
      embedding_dim=self.depth_emb_size
    ) 

    self.nt_embed_layer = nn.Embedding(
      num_embeddings=len(self.node_types),
      embedding_dim=self.nodetype_emb_size
    ) 

    rnn_input_size = [
      self.input_emb_size,
      self.depth_emb_size,
      self.nodetype_emb_size
    ]
    self.rnn = nn.GRU(
      input_size=sum(rnn_input_size),
      hidden_size=self.nhid,
      num_layers=self.num_layers,
      batch_first=True,
      dropout=self.dropout,
      bidirectional=self.bidirect
      )
    
    self.final_layer = nn.Linear(
      self.nhid,
      self.n_letters
    )


  def initHidden(self):
    if self.hid_init == 'zeros':
      return torch.zeros(self.num_layers, 1, self.nhid)
    elif self.hid_init == 'glorot_normal':
      w = torch.empty(self.num_layers, 1, self.nhid)
      return nn.init.xavier_normal_(w)

  def forward(self, node_type, depth, inputs, hidden):

    nt_out = self.nt_embed_layer(node_type)
    #Expand to match char out
    nt_out = nt_out.expand(
      inputs.size(0), 
      inputs.size(1), 
      self.nodetype_emb_size
      )

    depth_out = self.depth_embed_layer(depth)
    #Expand to match char out
    depth_out = depth_out.expand(
      inputs.size(0), 
      inputs.size(1), 
      self.depth_emb_size
      )

    input_out = self.input_embed_layer(inputs)


    combined_out = torch.cat([input_out, nt_out, depth_out], dim=2)
    
    out, hidden = self.rnn(combined_out, hidden)
    out = self.final_layer(out)
    #out = self.final_dropout(out)
    #out = self.softmax(out)
    #out = out.transpose(1,2)
    return out, hidden
  
   