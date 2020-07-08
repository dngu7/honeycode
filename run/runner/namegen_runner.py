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


class NamegenRunner(object):
  def __init__(self, config, model_config_branch, model_name='namegen'):

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

    self.end_token = self.model_config.dataset.end_token
    self.all_letters = list(string.printable) + [self.end_token]
    self.n_letters = len(self.all_letters) + 1 #EOS MARKER
    self.max_depth = self.model_config.dataset.max_depth
    self.max_child = self.model_config.dataset.max_child
    self.max_neigh = self.model_config.dataset.max_neigh
    self.node_types = self.model_config.dataset.node_types
    

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
    self.model_func = init_method(self.model_config, self.n_letters)
    
    #load checkpoints
    load_model(self.model_func, self.model_snapshot, self.device)

    self.ss_dist_file = os.path.join(self.model_dir, 'ss_dist.p')
    self.ss_dist = pickle.load(open(self.ss_dist_file, "rb"))

  def eval(self, req_nodes, req_max_depth):
    eval_time = time.time()
    
    self.model_func.to(self.device)
    self.model_func.eval() 
    #logger.debug("Starting on device={}".format(self.model_func.device))

    gen_depth = min(req_max_depth, self.max_depth)
    text_generated = {
      'dir': {i: [] for i in range(gen_depth + 1)}, 
      'file': {i: [] for i in range(1, gen_depth + 1)}, 
    }

    with torch.no_grad():
      for node_type in text_generated.keys():
        for depth in text_generated[node_type]:
          n_samples = 0
          while n_samples < req_nodes:
            #sample start letter from distributiion
            start_letter_dist = self.ss_dist[node_type][depth]
            #m = torch.distributions.Categorical(start_letter_dist)
            #start_letter_idx = m.sample()
            sample = np.random.multinomial(1, start_letter_dist, size=1)
            start_letter_idx = np.argmax(sample)

            input_eval = torch.LongTensor([start_letter_idx]).view(1, -1)
            nt_eval = torch.LongTensor([self.node_types.index(node_type)])
            depth_eval = torch.LongTensor([depth])

            #Move to gpu
            input_eval = input_eval.pin_memory().to(0,non_blocking=True)
            nt_eval = nt_eval.pin_memory().to(0,non_blocking=True)
            depth_eval = depth_eval.pin_memory().to(0,non_blocking=True)
            hidden = self.model_func.initHidden().pin_memory().to(0,non_blocking=True)

            text_gen = []
            for _ in range(self.max_gen_len):
              pred, hidden  = self.model_func(nt_eval, depth_eval, input_eval, hidden)

              pred = pred[0].squeeze()
              pred = pred / self.temperature
              m = torch.distributions.Categorical(logits=pred)
              pred_id = m.sample()

              #print(i, pred_id)
              next_char = self.all_letters[pred_id.item()]

              if next_char == self.end_token: break
              
              text_gen.append(next_char)
              input_eval = pred_id.view(-1,1)

            full_name = self.all_letters[start_letter_idx] + ''.join(text_gen)
            if full_name not in text_generated[node_type][depth]:
              text_generated[node_type][depth].append(full_name)
              n_samples += 1

    if self.save_sample:

      save_name = os.path.join(self.config.config_save_dir, 'sample_filenames_{}.txt'.format(
        str(int(time.time()))
      ))

      with open(save_name, 'w') as f:
        for node_type in text_generated.keys():
          f.write("-"*100)
          f.write("\nNode Type: {} \n".format(node_type))
          for depth in text_generated[node_type]:
            samples = text_generated[node_type][depth]
            out = ', '.join(samples)
            f.write("{:3} | {}\n".format(depth, out))
      
      logger.info("Saved sample @ {}".format(save_name))

    logger.debug("Generated {:2} filenames ({:2.2f} s)".format(req_nodes, time.time() - eval_time))
  
    return text_generated

