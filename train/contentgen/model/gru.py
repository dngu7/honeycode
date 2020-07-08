import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUCharModel(nn.Module):
  def __init__(self,config, n_letters, output_size):
    super(GRUCharModel, self).__init__()
    self.nhid = config.model.nhid
    self.char_emb_size = config.model.char_emb_size
    self.ext_emb_size = config.model.ext_emb_size
    self.num_layers = config.model.num_layers
    self.dropout = config.model.dropout
    self.bidirect = config.model.bidirectional
    self.n_letters = n_letters
    self.output_size = output_size
    self.input_size = n_letters
    self.hid_init = config.model.hid_init
    self.file_ext = config.dataset.file_ext
    self.seq_len = config.dataset.seq_len

    self.ext_embed_layer = nn.Embedding(
      num_embeddings=len(self.file_ext),
      embedding_dim=self.ext_emb_size
    ) 

    self.char_embed_layer = nn.Embedding(
      num_embeddings=self.n_letters,
      embedding_dim=self.char_emb_size
    ) 

    self.rnn = nn.GRU(
      input_size=self.char_emb_size + self.ext_emb_size,
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
    # self.final_dropout = nn.Dropout(0.1)
    # self.softmax = nn.LogSoftmax(dim=2)

  def initHidden(self):
    if self.hid_init == 'zeros':
      return torch.zeros(self.num_layers, 1, self.nhid)
    elif self.hid_init == 'glorot_normal':
      w = torch.empty(self.num_layers, 1, self.nhid)
      return nn.init.xavier_normal_(w)

  def forward(self, ext, inputs, hidden):

    ext_out = self.ext_embed_layer(ext)
    #Expand to match char out
    ext_out = ext_out.expand(
      inputs.size(0), 
      inputs.size(1), 
      self.ext_emb_size
      )

    char_out = self.char_embed_layer(inputs)
    combined_out = torch.cat([ext_out, char_out], dim=2)

    out, hidden = self.rnn(combined_out, hidden)
    out = self.final_layer(out)
    #out = self.final_dropout(out)
    #out = self.softmax(out)
    #out = out.transpose(1,2)
    return out, hidden
  
   