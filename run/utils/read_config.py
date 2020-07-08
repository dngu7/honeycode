import os
import yaml
import time
from easydict import EasyDict as edict
import numpy as np
import torch 

def read_config(config_file):
  return edict(yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader))

def get_config(config_file, output_dir=None, samples=1):

  # Load configuration
  config = edict(yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader))

  # Create configuration for saving
  config.run_id = str(os.getpid())
  config.exp_name = '_'.join([config.exp_name,
      time.strftime('%Y-%b-%d-%H-%M-%S'), config.run_id
  ])

  # Total number of repositories generated
  config.samples = samples

  # Save configuration in experiments folder
  config.config_save_dir = os.path.join(config.exp_dir, config.exp_name)
  save_name = os.path.join(config.config_save_dir, 'config.yaml')
  mkdir(config.exp_dir)
  mkdir(config.config_save_dir)
  yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)

  # Create output folder
  # Defaults to experiment folder if nothing specified
  if output_dir == None:
    config.save_dir = config.config_save_dir
  else:
    mkdir(output_dir)
    config.save_dir = output_dir

  #Seed and GPU
  np.random.seed(config.seed)
  torch.manual_seed(config.seed)
  torch.cuda.manual_seed_all(config.seed)

  config.use_gpu = config.use_gpu and torch.cuda.is_available()
  config.device = 'cuda:0' if config.use_gpu else 'cpu'

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
