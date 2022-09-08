import numpy as np
import random
from datetime import datetime
import json
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from tensorboardX import SummaryWriter
import os
# Note you can change the following to your own Visual Place Recognition module
from vpr.vpr_network import VPRNetwork
from vpr.uncertainty import Uncertainty
from vpr.dataset import create_safe_dataset, create_safe_dataloader
import vpr.utils as utils


class VPREvaluator(object):

    def __init__(self, config):
        self.config = config

        # Set random seed for reproductibility
        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"])
        torch.manual_seed(self.config["seed"])
        if not self.config["use_cpu"]:
            torch.cuda.manual_seed(self.config["seed"])

        # Read data
        self.evaluation_dataset = create_safe_dataset(self.config["evaluation_folder"])

        # Add network trainer
        self.device = torch.device("cuda" if not self.config["use_cpu"]
                                and torch.cuda.is_available() else "cpu")
        if self.config["vpr_head"] == "netvlad" or self.config[
                "vpr_head"] == "barebone" or self.config[
                "vpr_head"] == "gem":
            self.vpr_dnn = VPRNetwork(self.config, self.device)
        elif self.config["vpr_head"] == "uncertainty":
            self.vpr_dnn = Uncertainty(self.config, self.device)
        else:
            raise ValueError("Unknown VPR network")
        self.vpr_dnn.initialization()

    def evaluate(self):
        if self.config["resume_checkpoint"] == "":
            utils.evaluate(
                self.vpr_dnn, self.config["dataset_name"] + "_initial", self.evaluation_dataset, uncertainty=self.config["vpr_head"] == "uncertainty")
        else:
            utils.evaluate(
                self.vpr_dnn, self.config["dataset_name"], self.evaluation_dataset, uncertainty=self.config["vpr_head"] == "uncertainty")

    def run(self):
        self.evaluate()