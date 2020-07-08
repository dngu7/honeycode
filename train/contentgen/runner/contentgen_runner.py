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
import networkx as nx
import numpy as np
from tqdm import tqdm

from dataset.contentgen_char_dataset import *
from model.gru import GRUCharModel

from preprocess import graph_loader
from utils import data_parallel
from utils.train_helper import load_model, snapshot

import torch.optim as optim
import torch.nn as nn
 

logger = logging.getLogger('contentgen')


class ContentgenRunner(object):
  def __init__(self, config):
    logger.debug("contentgen runner initialized")

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
    self.num_test_gen = config.test.num_test_gen

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
    
    self.all_letters = list(string.printable) + ['<eof>']
    self.n_letters = len(self.all_letters) + 1 #EOS MARKER
    self.seq_len = config.dataset.seq_len
    self.file_ext = config.dataset.file_ext
    self.test_ext = config.test.test_ext
    self.test_nb = config.test.test_nb

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
      train_dataset.n_letters,
      train_dataset.seq_len
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
      iter_count = 0  
      for _, (inp, target, ext) in enumerate(train_iterator):
        
        model.module.zero_grad()
        optimizer.zero_grad()

        iter_count += 1
        loss = .0
         
        input_tensor = inp.pin_memory().to(0, non_blocking=True)          
        target_tensor = target.pin_memory().to(0, non_blocking=True)
        ext_tensor = ext.pin_memory().to(0, non_blocking=True)
        hidden = torch.cat([model.module.initHidden().pin_memory().to(0,non_blocking=True) for _ in range(input_tensor.size(0))], dim=1)

        output, hidden = model(ext_tensor, input_tensor, hidden)

        for batch in range(output.size(0)):
          l = criterion(output[batch], target_tensor[batch])
          loss += l
        avg_train_loss += float(loss.item()) / output.size(0)

        loss.backward()
        optimizer.step()
        #lr_scheduler.step()

        if iter_count % self.train_conf.display_iter == 0 and iter_count > 1:
          avg_train_loss /= self.train_conf.display_iter
          results['train_loss'] += [avg_train_loss]
          results['train_step'] += [iter_count]

          
          logger.info("Loss @ epoch {:04d} iteration {:08d} = {}".format(epoch + 1, iter_count, avg_train_loss))

          
      #if iter_count % self.train_conf.display_code_iter == 0 and iter_count > 0:
      #Look at only the first one
      choice = random.choice(range(output.size(0)))
      file_type = self.file_ext[ext_tensor[choice].squeeze().detach().item()]
      target_char = self.tochar(target_tensor[choice])
      predict_char = self.tochar(torch.argmax(output[choice], dim=1))
      logger.info("Epoch {} Iter {} | Sample Start ----------------------".format(epoch, iter_count))
      logger.info("File Type: {}".format(file_type))
      logger.info("Predict: {}".format(''.join(predict_char)))
      logger.info("Target : {}".format(''.join(target_char)))
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


  def eval(self):
    logger.debug('starting eval')
    
    self.config.save_dir = self.test_conf.test_model_dir

    model_file = os.path.join(self.config.save_dir, self.test_conf.test_model_name)
    batch_size = self.test_conf.test_batch_size
    num_gen = self.test_conf.num_test_gen
    start_string = self.test_conf.start_string
    temperature = self.test_conf.temperature

    assert batch_size == 1, "batch size needs to be one"

    logger.debug("Loading Model: {}".format(model_file))
    logger.debug("Batch Size: {} Num of characters to generate: {} Start string: {}".format(
      batch_size, num_gen, start_string
    ))

    model = eval(self.model_conf.model_name)(
      self.config,
      self.n_letters - 1,
      self.seq_len
      )

    model = load_model(
        model,
        model_file,
        self.device
        )['model']
      
    model = model.to(self.device)
    model.eval()
    
    text_generated = []
    input_eval = torch.LongTensor([self.all_letters.index(s) for s in start_string]).view(1, -1)
    ext = torch.LongTensor([self.file_ext.index(self.test_ext)])

    ext_tensor = ext.pin_memory().to(0, non_blocking=True)
    input_eval = input_eval.pin_memory().to(0,non_blocking=True)
    hidden = model.initHidden().pin_memory().to(0,non_blocking=True)
    
    with torch.no_grad():
      for i in range(num_gen):
        #print(i, input_eval.shape)
        pred, hidden  = model(ext_tensor, input_eval, hidden)
        pred = pred[0].squeeze()
        pred = pred / temperature
        
        m = torch.distributions.Categorical(logits=pred)
        pred_id = m.sample()
        if i == 0 and len(start_string) > 1:
            pred_id = pred_id[-1]


        #print(i, pred_id)
        next_char = self.all_letters[pred_id.item()]
        text_generated.append(next_char)

        input_eval = pred_id.view(-1,1)

    full_code = start_string + ''.join(text_generated)

    save_name = os.path.join(self.config.save_dir, 'sample_{}.{}'.format(self.test_nb, self.test_ext))

    with open(save_name, 'wb') as f:
      f.write(full_code.encode('utf-8'))


  