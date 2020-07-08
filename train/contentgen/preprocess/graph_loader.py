from preprocess import local_graph_loader
from os import path
import logging

logger = logging.getLogger('contentgen')


def load_graph(source='local', data_dir=None, min_num_nodes=5, max_num_nodes=5000, node_labels=True, graph_labels=True):
  assert source in ['local','db']
  assert path.exists(data_dir)

  # logger.debug("Graph Data Properties")
  # logger.debug("{:20}: {}".format("source", source))
  # logger.debug("{:20}: {}".format("data_dir", data_dir))
  # logger.debug("{:20}: {}".format("min_num_nodes", min_num_nodes))
  # logger.debug("{:20}: {}".format("max_num_nodes", max_num_nodes))
  # logger.debug("{:20}: {}".format("node_labels", node_labels))
  # logger.debug("{:20}: {}".format("graph_labels", graph_labels))
  logger.info("Loading Graphs...")

  if source == 'local':
    return local_graph_loader.load_graph(
      data_dir=data_dir, min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes, node_labels=node_labels, graph_labels=graph_labels
    )
  elif source == 'db':
    raise SystemError("Not implemented yet")
