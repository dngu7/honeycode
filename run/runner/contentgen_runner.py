import importlib.util
import logging
import os
import time
import random
import string
import pickle
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from utils.read_config import read_config
from utils.train_helper import load_model


logger = logging.getLogger('gen')


class ContentgenRunner(object):
  def __init__(self, config, model_config_branch, model_name='contentgen'):

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
    

    self.batch_size = 1
    self.temperature = model_config_branch.temperature
    self.max_gen_len = model_config_branch.max_gen_len
    self.save_sample = model_config_branch.save_sample

    self.file_exts = self.model_config.dataset.file_ext
    self.seq_len = self.model_config.dataset.seq_len
    self.end_token = self.model_config.dataset.end_token
    self.all_letters = list(string.printable) + [self.end_token]
    self.n_letters = len(self.all_letters) + 1 #EOS MARKER
    
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
    self.model_func = init_method(self.model_config, self.n_letters-1, self.seq_len)
    
    #load checkpoints
    load_model(self.model_func, self.model_snapshot, self.device)
    
  def tochar(self, tensor_idx):
    all_char = []
    for t in tensor_idx:
      t = t.squeeze().detach().item()
      all_char.append(self.all_letters[t])
    return all_char

  def eval(self, ext, start_string=None, name='contentgen'):
    eval_time = time.time()
    self.model_func.to(self.device)
    self.model_func.eval() 

    #start string is a random choice (except EOS)
    if start_string == None:
      start_string = self.random_gen.choice(self.all_letters[10:62] + ['#', ' '])

    text_generated = []
    ext_eval = torch.LongTensor([self.file_exts.index(ext)])
    ext_eval = ext_eval.pin_memory().to(0, non_blocking=True)

    input_eval = torch.LongTensor([self.all_letters.index(s) for s in start_string]).view(1, -1)
    input_eval = input_eval.pin_memory().to(0,non_blocking=True)
    hidden = self.model_func.initHidden().pin_memory().to(0,non_blocking=True)

    with torch.no_grad():
      for i in range(self.max_gen_len):
        pred, hidden  = self.model_func(ext_eval, input_eval, hidden)
        pred = pred[0].squeeze()
        pred = pred / self.temperature
        
        m = torch.distributions.Categorical(logits=pred)
        pred_id = m.sample()
        if i == 0 and len(start_string) > 1:
            pred_id = pred_id[-1]
        #print(i, pred_id)
        next_char = self.all_letters[pred_id.item()]
        text_generated.append(next_char)

        input_eval = pred_id.view(-1,1)

    full_string = start_string + ''.join(text_generated)
    full_string = full_string.encode('utf-8')

    if self.save_sample:
      save_name = os.path.join(self.config.config_save_dir, 'sample_{}.{}'.format(time.time(), ext))
      with open(save_name, 'wb') as f:
        f.write(full_string)

    logger.debug("Generated content for {} [{:2.2f} s]".format(name, time.time() - eval_time))
    #logger.debug("Generated {:2} filenames ({:.2f} s)".format(req_nodes, time.time() - eval_time))

    return full_string

