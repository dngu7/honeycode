import os
import sys
import click
from runner import repogen_runner
from utils import read_config, setup_logger



@click.command()
@click.option('--data','-d', default='julia', help='dataset language')
@click.option('--config_dir','-c', default='/config', help='Point to config folder')
@click.option('--log_level','-l', default='DEBUG')
@click.option('--device','-dv', default='gpu')
@click.option('--output_dir','-o', default=None)
@click.option('--samples','-s', default=1)
def main(data, config_dir, log_level, device, output_dir, samples):

  #Setup configuration files
  cur_dir = os.path.dirname(os.path.realpath(__file__))
  config_dir = cur_dir + config_dir
  if device == 'gpu':
    default_config = os.path.join(config_dir,'{}_gpu.yaml'.format(data))
  elif device == 'cpu':
    raise SystemError("Not implemented")
    
  config = read_config.get_config(default_config, output_dir, samples)

  #Setup Local Logging
  log_file = os.path.join(config.save_dir, "log_exp_{}.txt".format(config.run_id))
  logger = setup_logger.setup_logging(log_file, log_level)

  logger.info("Generating new {} repository @ {}".format(data, config.save_dir))

  runner = repogen_runner.RepoGenRunner(
    config=config
  )

  runner.eval()
  #invoke treegen to create graphs until there is one valid one


if __name__ == "__main__":
  main()