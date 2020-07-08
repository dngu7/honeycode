import logging
import os
import time
import random
import networkx as nx
from runner import trn_runner, namegen_runner, contentgen_runner

logger = logging.getLogger('gen')

class RepoGenRunner(object):
  def __init__(self, config):

    logger.debug("{} initialized".format(__name__))

    self.config = config
    self.use_gpu = config.use_gpu
    self.gpus = config.gpus
    self.device = config.device

    self.avail_ext_types = config.models.contentgen.ext
    self.seed = config.seed
    self.random_gen = random.Random(self.seed)

    self.start_strings = {
      'md': '#',
      'toml': '\n',
      'yml': None,
      'jl': '#',
      'txt': None,
      'REQUIRE': None
    }


  def eval(self):
    repo_gen_time = time.time()

    #####################################################
    ### Treegen
    runner = treegen_runner.TrnRunner(
      config=self.config,
      model_config_branch=self.config.models.treegen,
      model_name='treegen'
      )

    # Creates an arboresence with networkx 
    arboresences = runner.eval()
    
    all_max_nodes = [len(a.nodes) for a in arboresences]
    parent_nodes = []

    # Split the nodes into files, directories based on in and out edges
    all_node_details = []
    for a in arboresences:
      parent_node = None
      node_details = {}
      for n in a.nodes:
        inedges = a.in_edges(n)
        outedges = a.out_edges(n)
        #logger.debug("Node: {} inedges: {} outedges: {}".format(n, len(inedges), len(outedges)))
        if len(inedges) == 0:
          parent_node = n
        
        if len(outedges) == 0:
          node_details[n] = {'node_type': 'file'}
        else:
          node_details[n] = {'node_type': 'dir'}
      
      parent_nodes.append(parent_node)
      all_node_details.append(node_details)

      if parent_node == None:
        raise SystemError("Parent node not found")
    
    #Calculate depth of file
    all_max_depth = []
    for i, a in enumerate(arboresences):
      max_depth = 0
      for n in a.nodes:
        depth = 0
        if n != parent_nodes[i]:
          path = nx.shortest_path(a, parent_nodes[i], n)
          depth = len(path) - 1
        
        all_node_details[i][n]['depth'] = depth
        max_depth = max(max_depth, depth)
      
      all_max_depth.append(max_depth)
      #logger.debug("{} | {}".format(n, node_details[n]))


    #####################################################
    ### NAMEGEN

    runner = namegen_runner.NamegenRunner(
      config=self.config,
      model_config_branch=self.config.models.namegen,
      model_name='namegen'
      )

    for i, (max_nodes, max_depth) in enumerate(zip(all_max_nodes, all_max_depth)):
      #name for each arb
      names_generated = runner.eval(max_nodes, max_depth)
      
      gen_max_depth = max(names_generated['dir'].keys())

      name = '<NAME>'
      chosen_names = [name]

      #Rank by depth asc
      x = {n: all_node_details[i][n]['depth'] for n in all_node_details[i]}
      node_rank = {n: rank for rank, n in enumerate(sorted(x, key=x.get, reverse=False), 1)}
      for n in node_rank:
        depth = min(all_node_details[i][n]['depth'], gen_max_depth)
        node_type = all_node_details[i][n]['node_type']

        #choose random name without replacement
        #avail_names = names_generated[node_type][depth]
        #name = random.choice(avail_names)
        while name in chosen_names:
          if depth in [1]:
            name = names_generated[node_type][depth].pop(0)
          else:
            choice = self.random_gen.choice(range(len(names_generated[node_type][depth])))
            name = names_generated[node_type][depth].pop(choice)          

        all_node_details[i][n]['name'] = name
        chosen_names.append(name)

        #remove choice from list 
        #names_generated[node_type][depth].pop(avail_names.index(name))
        if node_type == 'file':
          if '.' in name:
            all_node_details[i][n]['ext'] = name.split('.')[-1]
      
      #logger.debug("Node {} | {}".format(n, node_details[n]))

    #####################################################
    ### Contentgen
    
    runner = contentgen_runner.ContentgenRunner(
      config=self.config,
      model_config_branch=self.config.models.contentgen,
      model_name='contentgen'
      )
    
    #Creates content for each node with an available extension type
    for i, node_details in enumerate(all_node_details):
      for n in node_details:
        if 'ext' in node_details[n]:
          ext = node_details[n]['ext']
          fn = node_details[n]['name']

          if ext in self.avail_ext_types:
            start_string = self.start_strings[ext]

            content = runner.eval(ext=ext, start_string=start_string, name=fn)
            all_node_details[i][n]['content'] = content
    
    #####################################################
    ### Generate an entire repository using BFS
    ### Start with parent node
    
    # Start with parent node
    for i, (arboresence, node_details) in enumerate(zip(arboresences, all_node_details)):
      stack = [(parent_nodes[i], self.config.save_dir)]
      finished_nodes = []
      
      while len(stack):
        cur_node, parent_path = stack.pop(0)

        #Get existing details
        node_detail = node_details[cur_node]

        #Create the creation path
        cur_path = os.path.join(parent_path, node_detail['name'])

        if node_detail['node_type'] == 'dir':
          if not os.path.exists(cur_path):
            os.mkdir(cur_path)
            logger.debug("DIR: {}".format(cur_path))

          #Add children to the stack if not already finished
          outedges = arboresence.out_edges(cur_node)
          for e in outedges:
            if e[1] not in finished_nodes:
              stack.append((e[1], cur_path))
        elif node_detail['node_type'] == 'file':
          content = ''.encode('utf-8')
          if 'content' in node_detail:
            content = node_detail['content']
            
          with open(cur_path, 'wb') as f:
            f.write(content)
          
          #logger.debug("FILE: {}".format(cur_path))
    
    logger.debug("Total Repogen Time: {:.2f}".format(time.time() - repo_gen_time))


