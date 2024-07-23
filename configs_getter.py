import inspect


import config


def get_configs():
  """Returns (config_name, config) tuples."""
  all_configs = [
      (name, conf) for name, conf in inspect.getmembers(config)
      if isinstance(conf, config._AlgosConfig) and not name.startswith('_')]
  
  return all_configs
 
