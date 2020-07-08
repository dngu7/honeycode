from preprocess import local_graph_loader
from os import path
import logging

logger = logging.getLogger('gen')

def load_graph(source='local', data_dir=None, min_num_nodes=5, max_num_nodes=5000, node_labels=True, graph_labels=True):
  assert source in ['local','db']
  assert path.exists(data_dir)

  logger.debug("Loading Graphs...")

  if source == 'local':
    return local_graph_loader.load_graph(
      data_dir=data_dir, min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes, node_labels=node_labels, graph_labels=graph_labels
    )
  elif source == 'db':
    raise SystemError("Not implemented yet")
    