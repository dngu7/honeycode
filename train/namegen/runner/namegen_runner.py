import copy
import logging
import os
import pickle
import sys
import time
import string
import random
from collections import defaultdict
import traceback

import numpy as np
from tqdm import tqdm

from dataset.namegen_char_dataset import *
from model.gru import GRUCharNameGenModel

from preprocess import graph_loader
from utils import data_parallel
from utils.train_helper import load_model, snapshot

import torch
import torch.optim as optim
import torch.nn as nn
 

logger = logging.getLogger('namegen')


class NamegenRunner(object):
  def __init__(self, config):
    logger.debug("namegen runner initialized")

    self.config = config
    self.seed = config.seed
    self.dataset_conf = config.dataset
    self.model_conf = config.model
    self.train_conf = config.train
    self.test_conf = config.test
    self.use_gpu = config.use_gpu
    self.gpus = config.gpus
    self.num_gpus = len(self.gpus)
    self.device = config.device
    self.is_shuffle = False
    self.batch_size = config.dataset.batch_size

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


    self.train_ratio = config.dataset.train_ratio
    self.dev_ratio = config.dataset.dev_ratio
    self.num_graphs = len(self.graphs)

    self.num_train = int(float(self.num_graphs) * self.train_ratio)
    self.num_dev = int(float(self.num_graphs) * self.dev_ratio)
    self.num_test_gt = self.num_graphs - self.num_train


    logger.info('Train/val/test = {}/{}/{}'.format(
      self.num_train, 
      self.num_dev,
      self.num_test_gt
      ))

    if self.is_shuffle:
      self.npr = np.random.RandomState(self.seed)
      self.npr.shuffle(self.graphs)

    self.graphs_train = self.graphs[:self.num_train]
    self.graphs_dev = self.graphs[:self.num_dev]
    self.graphs_test = self.graphs[self.num_train:]
    
    self.end_token = config.dataset.end_token
    self.all_letters = list(string.printable) + [self.end_token]
    self.n_letters = len(self.all_letters) + 1 #EOS MARKER
    self.max_depth = config.dataset.max_depth
    self.max_child = config.dataset.max_child
    self.max_neigh = config.dataset.max_neigh
    self.node_types = config.dataset.node_types


  def train(self):
    logger.debug('starting training')

    train_dataset = eval(self.dataset_conf.loader_name)(self.config, self.graphs_train, tag='train')
    
    #Get start and end tokens

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=self.batch_size,
        shuffle=self.train_conf.shuffle,
        num_workers=self.train_conf.num_workers,
        drop_last=False
        )

    model = eval(self.model_conf.model_name)(
      self.config,
      self.n_letters
      )

    #move to gpu and parallelize
    if self.use_gpu:
      model = data_parallel.DataParallel(model, device_ids=self.gpus).to(self.device)

    model_params = filter(lambda p: p.requires_grad, model.parameters())

    #Setup optimizer
    if self.train_conf.optimizer == 'SGD':
      optimizer = optim.SGD(
          model_params,
          lr=self.train_conf.lr,
          momentum=self.train_conf.momentum,
          weight_decay=self.train_conf.wd
          )      
    elif self.train_conf.optimizer == 'Adam':
      optimizer = optim.Adam(
        model_params, 
        lr=self.train_conf.lr, 
        weight_decay=self.train_conf.wd
        )
    else:
      raise ValueError("Non-supported optimizer!")

    # lr_scheduler = optim.lr_scheduler.MultiStepLR(
    #     optimizer,
    #     milestones=self.train_conf.lr_decay_epoch,
    #     gamma=self.train_conf.lr_decay)

    # reset gradient
    # for i, p in enumerate(model.parameters()):
    #     logger.info("{}: {}".format(i, p))
    # print("-"*80)
    #criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()

    # resume training
    resume_epoch = 0
    if self.train_conf.is_resume:
      resume_epoch = self.train_conf.resume_epoch
      model_file = os.path.join(self.train_conf.resume_dir,
                                self.train_conf.resume_model)
      obj = load_model(
          model.module if self.use_gpu else model,
          model_file,
          self.device,
          optimizer=optimizer
          )
      
      if self.use_gpu:
        model.module = obj['model']
      else:
        model = obj['model']

      optimizer = obj['optimizer']
      scheduler = obj['scheduler']

     
    results = defaultdict(list)
    
    for epoch in range(resume_epoch, self.train_conf.max_epoch):
      model.train()

      train_iterator = train_loader.__iter__()
      if epoch == 0:
        iter_length = len(train_iterator)
        logger.debug("Length of train loader: {}".format(iter_length))  

      avg_train_loss = .0
      loss = .0   
      iter_count = 0  
      for _, (inp, target, node_type, depth, n_child, n_neigh) in enumerate(train_iterator):
        model.module.zero_grad()
        optimizer.zero_grad()

        iter_count += 1
        
         
        inp = inp.pin_memory().to(0, non_blocking=True)          
        target_tensor = target.pin_memory().to(0, non_blocking=True)
        node_type = node_type.pin_memory().to(0, non_blocking=True)
        depth = depth.pin_memory().to(0, non_blocking=True)
        n_child = n_child.pin_memory().to(0, non_blocking=True)
        n_neigh = n_neigh.pin_memory().to(0, non_blocking=True)

        hidden = torch.cat([model.module.initHidden().pin_memory().to(0,non_blocking=True) for _ in range(inp.size(0))], dim=1)
        
        if epoch == 0 and iter_count == 1:
          logger.debug("inp shape:{}".format(inp.shape))
          logger.debug("target shape:{}".format(target_tensor.shape))
          logger.debug("node_type shape:{}".format(node_type.shape))
          logger.debug("depth shape:{}".format(depth.shape))
          logger.debug("hidden shape:{}".format(hidden.shape))

        output, hidden = model(node_type, depth, inp, hidden)

        l = criterion(output[0], target_tensor[0])
        loss += l
        avg_train_loss += float(l.item()) 

        if iter_count % 64 == 0:
          loss.backward()
          optimizer.step()
          loss = .0
          
        #lr_scheduler.step()

        if iter_count % self.train_conf.display_iter == 0 and iter_count > 1:
          avg_train_loss /= self.train_conf.display_iter
          results['train_loss'] += [avg_train_loss]
          results['train_step'] += [iter_count]


          
          logger.info("Loss @ epoch {:04d} iteration {:08d} = {}".format(epoch + 1, iter_count, avg_train_loss))
          
          
      #if iter_count % self.train_conf.display_code_iter == 0 and iter_count > 0:
      #Look at only the first one
      choice = 0
      pred_node_type = self.node_types[node_type[choice].squeeze().detach().item()]
      pred_depth = depth[choice].squeeze().detach().item()
      pred_target = self.tochar(target_tensor[choice])
      pred_output = self.tochar(torch.argmax(output[choice], dim=1))
      logger.info("Epoch {} Iter {} | Sample Start ----------------------".format(epoch + 1, iter_count))
      logger.info("Node Type: {} , Depth: {}".format(pred_node_type, pred_depth))
      logger.info("Predict: {}".format(''.join(pred_target)))
      logger.info("Target : {}".format(''.join(pred_output)))
      logger.info("--------------------------------------------------------")
      #logger.info("output: {}".format(output[0]))

        # snapshot model
      if epoch % self.train_conf.snapshot_epoch == 0:
        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
        
        snapshot(model.module, optimizer, self.config, epoch + 1)
        

    pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))

    
    return 1

  def tochar(self, tensor_idx):
    all_char = []
    for t in tensor_idx:
      t = t.squeeze().detach().item()
      all_char.append(self.all_letters[t])
    return all_char

  def start_string_dist(self):
    '''Obtains the start string distribution'''
    

    ss_dist = {
      'dir': {i: [0]*self.n_letters for i in range(self.max_depth + 1)}, 
      'file': {i: [0]*self.n_letters for i in range(self.max_depth + 1)}, 
    }

    logger.debug("Calculating start letter distribution")

    for graph in tqdm(self.graphs, total=len(self.graphs)):
      for n in graph.nodes.data():
        node_type = n[1]['node_type'].decode("utf-8")
        path_split = n[1]['node_path'].decode("utf-8").split('/')
        depth = min(self.max_depth, len(path_split) - 2)
        filename = path_split[-1]
        ss_idx = self.all_letters.index(filename[0])
        
        ss_dist[node_type][depth][ss_idx] += 1
    
    #normalize
    for node_type in ss_dist.keys():
      for depth in ss_dist[node_type].keys():
        counts = ss_dist[node_type][depth]
        total_sum = sum(counts)
        #Reduce
        ss_dist[node_type][depth] = np.array(counts) / total_sum

    return ss_dist
    
  def eval(self):
    ## This should generate a sample for every single type

    logger.debug('starting eval')
    
    self.config.save_dir = os.path.join(self.test_conf.test_exp_dir, self.test_conf.test_model_dir)

    model_file = os.path.join(self.config.save_dir, self.test_conf.test_model_name)
    ss_dist_file = os.path.join(self.config.save_dir, 'ss_dist.p')
    batch_size = self.test_conf.test_batch_size
    max_gen_length = self.test_conf.max_gen_length
    num_gen_samples = self.test_conf.num_gen_samples
    #start_string = self.test_conf.start_string
    temperature = self.test_conf.temperature

    #Calc the start string dist for each kind of file
    if not os.path.exists(ss_dist_file): 
      ss_dist = self.start_string_dist()
      pickle.dump(ss_dist, open(ss_dist_file, "wb"))
    else:
      ss_dist = pickle.load(open(ss_dist_file, "rb"))

    assert batch_size == 1, "batch size needs to be one"

    logger.debug("Loading Model: {}".format(model_file))
    logger.debug("Batch Size: {} max_length: {}".format(
      batch_size, max_gen_length
    ))

    model = eval(self.model_conf.model_name)(
      self.config,
      self.n_letters
      )

    model = load_model(
        model,
        model_file,
        self.device
        )['model']
      
    model = model.to(self.device)
    model.eval()
    
    text_generated = {
      'dir': {i: [] for i in range(self.max_depth + 1)}, 
      'file': {i: [] for i in range(1, self.max_depth + 1)}, 
    }

    with torch.no_grad():
      for node_type in text_generated.keys():
        for depth in text_generated[node_type]:
          n_samples = 0
          while n_samples < num_gen_samples:
            #sample start letter from distributiion
            start_letter_dist = ss_dist[node_type][depth]
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
            hidden = model.initHidden().pin_memory().to(0,non_blocking=True)

            text_gen = []
            for _ in range(max_gen_length):
              pred, hidden  = model(nt_eval, depth_eval, input_eval, hidden)

              pred = pred[0].squeeze()
              pred = pred / temperature
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
              

    save_name = os.path.join(self.config.save_dir, 'name_gen_sample_{}.txt'.format(
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


  
