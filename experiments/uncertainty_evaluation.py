from vpr.vpr_evaluator import VPREvaluator
from vpr.utils import load_config

config_file = "config/evaluation_uncertainty.yml"
config = load_config(config_file)
evaluator = VPREvaluator(config)
print("Setup completed.")
evaluator.run()
print("Evaluation completed.")