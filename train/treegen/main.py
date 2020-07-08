import logging
import logging.config
import os
import sys
import click
from runner import trn_runner
from utils import read_config, setup_logger


@click.command()
@click.option('--network','-n', default='trn', help='Choose your neural net')
@click.option('--mode','-m', default='train',
  type=click.Choice(['train', 'eval'], case_sensitive=False),
  help='Train or evaluation mode')
@click.option('--data','-d', default='julia', help='Dataset')
@click.option('--config_dir','-c', default='/config', help='Point to config folder')
@click.option('--log_level','-l', default='DEBUG')

def main(network, mode, data, config_dir, log_level):
  # Setup Config (if exists)
  cur_dir = os.path.dirname(os.path.realpath(__file__))
  config_dir = cur_dir + config_dir
  default_config = os.path.join(config_dir,'{}_{}.yaml'.format(network, data))
  config = read_config.get_config(default_config, is_test=True if mode == 'eval' else False)

  # Setup Local Logging
  log_file = os.path.join(config.save_dir, "log_exp_{}.txt".format(config.run_id))
  logger = setup_logger.setup_logging(log_file, log_level)

  
  if network == 'trn':
    runner = trn_runner.TrnRunner(
      config=config
      )

  if mode == 'train':
    runner.train()
  elif mode =='eval':
    runner.eval()



if __name__ == "__main__":
  main()
