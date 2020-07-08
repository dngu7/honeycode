import os
import yaml
import torch
from easydict import EasyDict as edict


def edict2dict(edict_obj):
  dict_obj = {}

  for key, vals in edict_obj.items():
    if isinstance(vals, edict):
      dict_obj[key] = edict2dict(vals)
    else:
      dict_obj[key] = vals

  return dict_obj


def snapshot(model, optimizer, config, step, gpus=[0], tag=None, scheduler=None):
  
  if scheduler is not None:
    model_snapshot = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step
    }
  else: 
    model_snapshot = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),        
        "step": step
    }    

  torch.save(model_snapshot,
             os.path.join(config.save_dir, "model_snapshot_{}.pth".format(tag)
                          if tag is not None else
                          "model_snapshot_{:07d}.pth".format(step)))
  # update config file's test path
  save_name = os.path.join(config.save_dir, 'config.yaml')
  # config_save = edict(yaml.load(open(save_name, 'r'), Loader=yaml.FullLoader))
  config_save = edict(yaml.load(open(save_name, 'r')))
  config_save.test.test_model_dir = config.save_dir
  config_save.test.test_model_name = "model_snapshot_{}.pth".format(
          tag) if tag is not None else "model_snapshot_{:07d}.pth".format(step)
          
  yaml.dump(edict2dict(config_save), open(save_name, 'w'), default_flow_style=False)


def load_model(model, file_name, device, optimizer=None, scheduler=None):
  model_snapshot = torch.load(file_name, map_location=device)  
  model.load_state_dict(model_snapshot["model"])
  if optimizer is not None:
    optimizer.load_state_dict(model_snapshot["optimizer"])

  if scheduler is not None:
    scheduler.load_state_dict(model_snapshot["scheduler"])

