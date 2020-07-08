import logging
import sys
import os
import pickle
import time
import glob
import re
import shutil
import numpy as np
import torch
from tqdm import tqdm
import unicodedata
import traceback
import string

logger = logging.getLogger('contentgen')


class ContentgenCharDataset(object):
  def __init__(self, config, graphs, tag):
    self.config = config
    self.device = config.device
    self.cpucores = config.cpucores
    self.tag = tag
    self.dataset_name = config.dataset.dataset_name
    self.data_path = config.dataset.data_path
    self.repo_path = config.dataset.repo_path
    
    self.language = config.dataset.language
    self.all_letters = string.printable
    self.n_letters = len(self.all_letters) + 1 #EOS MARKER
    self.seq_len = config.dataset.seq_len
    self.window_shift_len = config.dataset.window_shift_len
    
    self.start_token = config.dataset.start_token
    self.end_token = config.dataset.end_token
    
    #self.tokendict_conf = config.tokendict

    self.model_name = config.model.model_name
    
    ## SETTINGS
    self.limit_precompute_repos = config.dataset.limit_precompute_repos
    self.limit_precompute_inputs = config.dataset.limit_precompute_inputs

    self.limit_repos = config.dataset.limit_repos
    self.file_ext = config.dataset.file_ext 
    self.input_func = self._precompute_inputs_v1

    self.batch_size = config.dataset.batch_size

    self.graphs = graphs
    self.num_graphs = len(graphs)
    self.is_overwrite_precompute_repos = config.dataset.is_overwrite_precompute_repos
    self.is_overwrite_precompute_inputs = config.dataset.is_overwrite_precompute_inputs
    
    self.save_path = os.path.join(
        self.data_path, '{}_precompute'.format(
            self.dataset_name))
    
    self.save_dir_one = os.path.join(self.save_path, '{}_repos'.format(tag))
    self.save_dir_two = os.path.join(self.save_path, '{}_inputs'.format(tag))
    
    for f in [self.save_path, self.save_dir_one, self.save_dir_two]:
      if not os.path.exists(f):
        os.mkdir(f)
        logger.info("Created new dir: {}".format(f))
  
    
    if self.is_overwrite_precompute_repos:
      ans = input('Choose overwrite mode. A: Delete and overwrite, B: Overwrite as you go (A/B)')
      if ans == 'A':
        shutil.rmtree(self.save_dir_one)
        os.mkdir(self.save_dir_one)
      elif ans != 'B':
        sys.exit(1)
      self.start_precompute_chars()


    self.precompute_repos = glob.glob(os.path.join(self.save_dir_one, '*.p'))

    if len(self.precompute_repos) == 0:
      raise ValueError("No repos found in {}".format(self.save_dir_one))
    
    logger.info("Available Extensions: {}".format(', '.join(self.file_ext)))

    if self.is_overwrite_precompute_inputs:
      shutil.rmtree(self.save_dir_two)
      os.mkdir(self.save_dir_two)
      self.start_precompute_inputs()

    self.precomputed_inputs = glob.glob(os.path.join(self.save_dir_two, '*.p'))
    logger.debug("Total file count: {}".format(len(self.precomputed_inputs)))
  

  def start_precompute_chars(self):
    logger.info("Precomputing chars - singlethread mode")
    self._singlethread_precompute_chars()
  
  def _singlethread_precompute_chars(self):
    for index, graph in enumerate(self.graphs):
      self.precompute_chars(graph, index)

  def precompute_chars(self, graph, index):
    #start_time = time.time()
    stars = graph.graph['stars']
    name = graph.graph['tree_name']
    if stars < 250:
      return 0

    valid_files = []
    file_dict = {}
    for n in graph.nodes.data():
      #print(name, stars, stars < 250, n[0])
      if n[1]['node_type'] == b'file':
        path = n[1]['node_path'].decode("utf-8")
        path_split = path.split('/')
        #print(path_split)
        filename = path_split[-1]
        parent_dir = path_split[1]
        direct_parent_dir = path_split[-2]
        depth = len(path_split) - 2 #minus parent and empty
        

        if '.' in filename: 
          ext = n[1]['node_path'].decode("utf-8").split('.')[-1]
        elif filename in ['REQUIRE']:
          ext = 'REQUIRE'
        else:
          #skip files without extensions
          continue 
        

        if self.limit_precompute_repos:
          if not any(r.lower() in parent_dir.lower() for r in self.limit_repos): 
            #this limits julia files
            return
             

        if ext in ['gif', 'jpg', 'png', 'pdf', 'jpeg', 'pptx']:
          #Images not saved
          continue
      
        full_path = self.repo_path + path
        if not os.path.exists(self.repo_path):
          raise SystemError("Repo path does not exist")

        if os.path.exists(full_path):
          all_chars = []
          try:
            with open(full_path, 'r', encoding='utf-8') as f:
              l = f.read()
              all_chars = list(l)
                

          except UnicodeDecodeError:
            logger.error("Source: {} Error: {}".format(full_path, traceback.format_exc()))
            continue
          
          file_dict = {}
          file_dict['all_chars'] = all_chars
          file_dict['path'] = path
          file_dict['filename'] = filename
          file_dict['parent_dir'] = parent_dir
          file_dict['d_parent_dir'] = direct_parent_dir
          file_dict['depth'] = depth
          file_dict['ext'] = ext

          valid_files.append(file_dict)

    #final Steps
    if valid_files:
      save_name = os.path.join(self.save_dir_one, '{}.p'.format(parent_dir))
      pickle.dump(valid_files, open(save_name, 'wb'))
      logger.debug("saved: {}".format(save_name))

  def unicodeToAscii(self, s):
    '''From pytorch tutorial'''
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in self.all_letters
    )

  def start_precompute_inputs(self):    
    logger.info("Precomputing chars - singlethread mode")
    self._singlethread_precompute_inputs()
  
  def _singlethread_precompute_inputs(self):
    #Computes with only julia files
    #Inputs: 
    start = time.time()
    for repo_idx, repo_path in tqdm(enumerate(self.precompute_repos)):
      self.input_func(repo_path, repo_idx)
    logger.info("singlethread time: {:.3f} s".format(time.time() - start))

  def _precompute_inputs_v1(self, repo_path, repo_idx):
    #Level 1: Only julia as the extension
    repo_files = pickle.load(open(repo_path, "rb"))
    count = 0
    for file_idx, file_dict in enumerate(repo_files):
        
        filename = file_dict['filename']
        d_parent_dir = file_dict['d_parent_dir']
        parent_dir = file_dict['parent_dir']

 
        ext = file_dict['ext']
        if ext not in self.file_ext: continue
        if 'test' in d_parent_dir.lower(): continue
        if 'example' in d_parent_dir.lower(): continue

        #If not in limit repo, then must be file extension
        '''List of filters'''
        if self.limit_precompute_inputs:
          if not any(r.lower() in parent_dir.lower() for r in self.limit_repos): 
            #This increases the number of non julia files
            #if ext not in ['toml', 'yml', 'REQUIRE','bib', 'md']:
            #  return
            #this limits julia files
            return
            #if ext not in self.file_ext[1:]:
            #  continue

        #The only feature is the indexes
        all_chars = file_dict['all_chars']
        all_ascii = [self.unicodeToAscii(c) for c in all_chars]
        all_idx = [self.all_letters.find(li) for li in all_ascii]
        all_idx_lbl = all_idx[1:] + [self.n_letters - 1]
        nwindows = len(all_ascii) // self.window_shift_len
        window_start = 0
        for w in range(nwindows):
          window_end = window_start + self.seq_len
          #window_ascii = all_ascii[window_start:window_end]
          window_idx = all_idx[window_start:window_end]
          label = all_idx_lbl[window_start:window_end] #shift to right
          window_start += self.window_shift_len

          if len(window_idx) == self.seq_len:
          
            input_dict = {
              #'ascii': window_ascii,
              'inp': window_idx,
              'label': label,
              'ext': [self.file_ext.index(ext)],
              'd_parent_dir': d_parent_dir,
              'depth': file_dict['depth'],
              'filename': file_dict['filename']
            }

            input_file = '{}_{}-{}-{}-{}.p'.format(self.tag, repo_idx, file_idx, w, ext)
            new_path = os.path.join(self.save_dir_two, input_file)
            pickle.dump(input_dict, open(new_path, 'wb'))




  def __len__(self):
    return len(self.precomputed_inputs)

  def inputTensor(self, line):
    return torch.LongTensor([self.all_letters.find(li) for li in line])
  
  def targetTensor(self, line):
    letter_indexes = [self.all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(self.n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

  # One-hot vector for extensions
  def extTensor(self, ext):
      li = self.file_ext.index(ext)
      tensor = torch.zeros(1, len(self.file_ext))
      tensor[0][li] = 1
      return tensor

  def __getitem__(self, idx):
    '''
    #Output shape = [Batch Size, Seq Length and Vocab Size]
      input_dict = {
        'ascii': window_ascii,
        'inp': window_idx,
        'label': label,
        'ext': ext,
        'd_parent_dir': d_parent_dir,
        'depth': file_dict['depth'],
        'filename': file_dict['filename']
      }
    '''

    input_file = self.precomputed_inputs[idx]
    with open(input_file, 'rb') as f:
      input_dict = pickle.load(f)

    ext_tensor = torch.LongTensor(input_dict['ext'])

    input_line_tensor = torch.LongTensor(input_dict['inp'])
    target_line_tensor = torch.LongTensor(input_dict['label'])
    #print("fn", input_dict['filename'], "inp:", len(input_dict['inp']), "lbl", len(input_dict['label']))

    return (input_line_tensor, target_line_tensor, ext_tensor)


