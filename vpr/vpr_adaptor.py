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
from vpr.vpr_trainer import VPRTrainer
from vpr.uncertainty_trainer import UncertaintyTrainer
from vpr.dataset import create_safe_dataset, create_safe_dataloader
import vpr.utils as utils

class VPRAdaptator(object):

    def __init__(self, config):
        self.config = config

        # Set random seed for reproductibility
        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"])
        torch.manual_seed(self.config["seed"])
        if not self.config["use_cpu"]:
            torch.cuda.manual_seed(self.config["seed"])

        # Read data
        self.dataset = create_safe_dataset(self.config["samples_folder"])
        self.evaluation_dataset = create_safe_dataset(self.config["evaluation_folder"])
        self.dataloader = create_safe_dataloader(self.dataset)
        self.read_index_files()

        # Add network trainer
        if self.config["vpr_head"] == "netvlad" or self.config[
                "vpr_head"] == "barebone" or self.config[
                "vpr_head"] == "gem":
            self.trainer = VPRTrainer(self.config, self.dataset)
        elif self.config["vpr_head"] == "uncertainty":
            self.trainer = UncertaintyTrainer(self.config)
        else:
            raise ValueError("Unknown VPR network")

    def load_config(self):
        with open(self.config_file, "r") as stream:
            return yaml.safe_load(stream)

    def read_index_files(self):
        # Read tuning files
        with open(os.path.join(self.config["samples_folder"], "inliers.txt"),
                  "r") as f:
            lines = f.readlines()
            self.inliers = np.zeros((len(lines), 2))
            for i in range(len(lines)):
                idx = lines[i].split(" ")
                self.inliers[i][:] = np.int64(idx[0]), np.int64(idx[1])

        with open(os.path.join(self.config["samples_folder"], "outliers.txt"),
                  "r") as f:
            lines = f.readlines()
            self.outliers = np.zeros((len(lines), 2))
            for i in range(len(lines)):
                idx = lines[i].split(" ")
                self.outliers[i][:] = np.int64(idx[0]), np.int64(idx[1])

        with open(os.path.join(self.config["samples_folder"], "triplets.txt"),
                  "r") as f:
            lines = f.readlines()
            self.training_tuples = []
            for i in range(len(lines)):
                idx = lines[i].split(" ")
                self.training_tuples.append([int(e) for e in idx])

        # Filter out similar inliers to avoid overfitting on parts of the trajectory
        inlier_max_proximity = 0
        filtered_inliers = []
        filtered_inliers.append(
            self.inliers[0]) 
        for i in range(1, len(self.inliers)):
            if abs(self.inliers[i][0] -
                   filtered_inliers[-1][0]) > inlier_max_proximity:
                filtered_inliers.append(self.inliers[i])
        self.inliers = np.asarray(filtered_inliers)

        # Sort out the tuples
        self.triplets = []
        self.training_info = {}
        for k in range(len(self.inliers)):
            anchor_id = int(self.inliers[k, 0])
            positive_id = int(self.inliers[k, 1])

            negatives_id = []
            for i in range(len(self.training_tuples)):
                if (self.training_tuples[i][0] == anchor_id
                        and self.training_tuples[i][1] == positive_id) or (
                            self.training_tuples[i][1] == anchor_id
                            and self.training_tuples[i][0] == positive_id):
                    for t in self.training_tuples[i][2:]:
                        if abs(t - anchor_id) > self.config[
                                "min_tuned_image_distance"] and abs(
                                    t - positive_id
                                ) > self.config["min_tuned_image_distance"]:
                            found = False
                            for j in range(len(self.inliers)):
                                if abs(self.inliers[j, 0] - anchor_id
                                       ) < 2 or abs(self.inliers[j, 0] -
                                                    positive_id) < 2:
                                    if abs(self.inliers[j, 1] -
                                           t) < self.config[
                                               "min_tuned_image_distance"]:
                                        found = True
                                elif abs(self.inliers[j, 1] - anchor_id
                                         ) < 2 or abs(self.inliers[j, 1] -
                                                      positive_id) < 2:
                                    if abs(self.inliers[j, 0] -
                                           t) < self.config[
                                               "min_tuned_image_distance"]:
                                        found = True
                            if not found:
                                negatives_id.append(t)

            nb_negatives = len(negatives_id)

            # Train on max 10 negatives
            if nb_negatives > 10:
                negatives_id = negatives_id[:10]
                nb_negatives = len(negatives_id)

            if nb_negatives == 0:
                continue

            self.triplets.append({
                "anchor_id": anchor_id,
                "positive_id": positive_id,
                "negatives_id": negatives_id,
            })

        self.training_info["inliers"] = self.triplets
        self.training_info["outliers"] = self.outliers.tolist()

        with open(
                os.path.join(self.config["samples_folder"],
                             "filtered_triplets.txt"), "w") as f:
            for t in self.triplets:
                f.write("{} {} {}\n".format(
                    str(t["anchor_id"]), str(t["positive_id"]),
                    " ".join([str(e) for e in t["negatives_id"]])))

    def adapt(self):
        print("Adapting model")
        writer = SummaryWriter(log_dir=os.path.join(
            self.config["models_path"],
            datetime.now().strftime("%b%d_%H-%M-%S") + "_" +
            self.config["vpr_head"] + "_" + self.config["backbone_model"]))

        # Write checkpoints in logdir
        logdir = writer.file_writer.get_logdir()
        self.save_path = os.path.join(logdir, "checkpoints")

        os.makedirs(self.save_path)

        self.evaluate("initial")
        for epoch in range(self.config["nb_epochs"]):
            self.trainer.scheduler_step(epoch)
            random.shuffle(self.training_info["inliers"])
            random.shuffle(self.training_info["outliers"])
            epoch_loss = self.trainer.train_triplets(epoch, self.training_info,
                                                     self.dataset)
            
            print("Epoch {} Complete: Loss: {:.4f}".format(
                epoch, epoch_loss),
                  flush=True)
            writer.add_scalar("Train/Loss", epoch_loss, epoch)

            utils.save_progress(self.trainer, epoch, self.save_path)

            self.evaluate(epoch)

        writer.close()

    def evaluate(self, epoch):
        if self.config["evaluate_on_other_dataset"]:
            utils.evaluate(
                    self.trainer.vpr_dnn, str(epoch), self.evaluation_dataset, uncertainty=self.config["vpr_head"] == "uncertainty")

    def run(self):
        self.adapt()