# All the NetVLAD code is heavily based on the following implementation:
# https://github.com/Nanne/pytorch-NetVlad

# If you want to use another VPR network, make sure to implement
# at least the methods called in vpr_trainer.py.

from audioop import avg
from math import log10, ceil, isnan
import random, shutil, json
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset


# To remove predatory warnings from scikit-learn
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import h5py
import faiss

import numpy as np
from vpr.uncertainty import Uncertainty
#from vpr.gem import ImageRetrievalNet
from vpr.bayesian_triplet_loss import BayesianTripletLoss, KLDivergenceLoss
import glob


class UncertaintyTrainer(object):

    def __init__(self, config):
        self.config = config

        self.device = torch.device("cuda" if not self.config["use_cpu"]
                                   and torch.cuda.is_available() else "cpu")

        self.vpr_dnn = Uncertainty(self.config, self.device) 

        self.vpr_dnn.initialization()

        # Set optimizer
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                          self.vpr_dnn.model.parameters()),
                                   lr=self.config["lr"],
                                   momentum=self.config["momentum"],
                                   weight_decay=self.config["weight_decay"])

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config["lr_step"],
            gamma=self.config["lr_gamma"])

        self.var_prior = 1 / self.vpr_dnn.encoder_dim
        self.triplet_criterion = BayesianTripletLoss(
            margin=self.config["margin"], var_prior=self.var_prior)
        self.triplet_criterion.to(self.device)

        self.outliers_kl_criterion = KLDivergenceLoss(var_prior=self.var_prior*10)
        self.outliers_kl_criterion.to(self.device)

    def scheduler_step(self, epoch):
        self.scheduler.step(epoch)

    def train_triplets(self, epoch, training_info, dataset):
        epoch_loss = 0.0
        batch_loss = 0.0

        triplets = training_info["inliers"]

        print("Training epoch " + str(epoch) + " Number of samples = " +
              str(len(triplets)))

        self.vpr_dnn.model.train()

        for iteration in range(len(triplets)):
            # Calculate loss for each Query, Positive, Negative triplet
            # due to potential difference in number of negatives have to
            # do it per query, per negative
            triplet = triplets[iteration]
            anchor_id = int(triplet["anchor_id"])  
            positive_id = int(triplet["positive_id"]) 
            negatives_id = triplet["negatives_id"]
            if anchor_id not in dataset or positive_id not in dataset or len(
                    negatives_id) == 0:
                print("Skipping triplet " + str(anchor_id) + " " +
                      str(positive_id))
                continue
            mean_anchor_encoding, var_anchor_encoding = self.vpr_dnn.encode(
                dataset[anchor_id][0])
            mean_positive_encoding, var_positive_encoding = self.vpr_dnn.encode(
                dataset[positive_id][0])

            loss = 0
            for n in range(len(negatives_id)):
                negative_id = int(negatives_id[n]) 

                if negative_id not in dataset:
                    continue

                mean_negative_encoding, var_negative_encoding = self.vpr_dnn.encode(
                    dataset[negative_id][0])

                loss += self.triplet_criterion(mean_anchor_encoding,
                                               var_anchor_encoding,
                                               mean_positive_encoding,
                                               var_positive_encoding,
                                               mean_negative_encoding,
                                               var_negative_encoding)

                del mean_negative_encoding, var_negative_encoding
            del mean_anchor_encoding, mean_positive_encoding, var_anchor_encoding, var_positive_encoding

            loss /= len(
                negatives_id)  # Normalise by actual number of negatives

            loss.backward()

            if (iteration + 1) % self.config[
                    "batch_size"] == 0 or iteration >= len(triplets) - 1:
                print(
                    "Batch optimization. Progress ({}/{}) Loss {:.4f}".format(
                        iteration+1, len(triplets), batch_loss))
                self.optimizer.step()
                self.optimizer.zero_grad()
                batch_loss = 0.0

            batch_loss += loss.item()
            epoch_loss += loss.item()
            
        if self.config["use_outliers"]:
            for iteration in range(len(training_info["outliers"])):
                anchor_id = int(training_info["outliers"][iteration][0])
                negative_id = int(training_info["outliers"][iteration][1])
                if anchor_id not in dataset or negative_id not in dataset:
                    continue
                mean_anchor_encoding, var_anchor_encoding = self.vpr_dnn.encode(
                    dataset[anchor_id][0])
                mean_negative_encoding, var_negative_encoding = self.vpr_dnn.encode(
                    dataset[negative_id][0])

                loss = 0
                loss += self.outliers_kl_criterion(var_anchor_encoding)
                loss += self.outliers_kl_criterion(var_negative_encoding)
                loss /= 2

                del mean_anchor_encoding, mean_negative_encoding, var_anchor_encoding, var_negative_encoding

                loss.backward()

                if (iteration +
                        1) % self.config["batch_size"] == 0 or iteration >= len(
                            training_info["outliers"]) - 1:
                    print(
                        "Batch optimization. Progress ({}/{}) Loss {:.4f}".format(
                            iteration+1, len(training_info["outliers"]), batch_loss))
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    batch_loss = 0.0

                batch_loss += loss.item()
                epoch_loss += loss.item()

        del loss
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()

        return epoch_loss
