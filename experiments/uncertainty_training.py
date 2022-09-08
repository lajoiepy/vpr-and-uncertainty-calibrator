from vpr.vpr_adaptor import VPRAdaptator
from vpr.utils import load_config

config_file = "config/adapt_uncertainty.yml"
config = load_config(config_file)
adaptator = VPRAdaptator(config)
print("Setup completed.")
adaptator.run()
print("Adaptation completed.")