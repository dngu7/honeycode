import os
import yaml
import time
import argparse
from easydict import EasyDict as edict
import numpy as np
import torch 

def get_config(config_file, exp_dir=None, is_test=False):
  """ Construct and snapshot hyper parameters """
  # config = edict(yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader))
  config = edict(yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader))

  # create hyper parameters
  config.run_id = str(os.getpid())
  config.exp_name = '_'.join([
      config.model.model_name, config.dataset.dataset_name,
      time.strftime('%Y-%b-%d-%H-%M-%S'), config.run_id
  ])


  if config.train.is_resume and not is_test:
    config.save_dir = config.train.resume_dir
    save_name = os.path.join(config.save_dir, 'config_resume_{}.yaml'.format(config.run_id))  
  else:    
    config.save_dir = os.path.join(config.exp_dir, config.exp_name)
    save_name = os.path.join(config.save_dir, 'config.yaml')

  if not is_test:
    mkdir(config.exp_dir)
    mkdir(config.save_dir)
    yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)
  else:
    config.save_dir = config.test.test_model_dir


  #Seed and GPU
  np.random.seed(config.seed)
  torch.manual_seed(config.seed)
  torch.cuda.manual_seed_all(config.seed)
  config.use_gpu = config.use_gpu and torch.cuda.is_available()

  return config


def edict2dict(edict_obj):
  dict_obj = {}

  for key, vals in edict_obj.items():
    if isinstance(vals, edict):
      dict_obj[key] = edict2dict(vals)
    else:
      dict_obj[key] = vals

  return dict_obj


def mkdir(folder):
  if not os.path.isdir(folder):
    os.makedirs(folder)