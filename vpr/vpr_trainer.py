# All the NetVLAD code is heavily based on the following implementation:
# https://github.com/Nanne/pytorch-NetVlad

# If you want to use another VPR network, make sure to implement
# at least the methods called in vpr_trainer.py.

from audioop import avg
from math import log10, ceil
import random, shutil, json
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ


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
from vpr.vpr_network import VPRNetwork
import glob

class VPRTrainer(object):

    def __init__(self, config, initialization_dataset=None):
        self.config = config

        self.device = torch.device("cuda" if not self.config["use_cpu"]
                                   and torch.cuda.is_available() else "cpu")

        self.vpr_dnn = VPRNetwork(self.config, self.device)

        if self.config["vpr_head"] == "netvlad":
            self.centroids_file = join(
                self.config["models_path"], "centroids",
                self.config["backbone_model"] + "_" +
                str(self.vpr_dnn.nb_clusters) + "_desc_cen.hdf5")
            if self.config["resume_checkpoint"] == "" and not exists(self.centroids_file):
                if initialization_dataset is None:
                    raise ValueError(
                        "No initialization dataset provided to compute centroids")
                self.compute_clusters(initialization_dataset)

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

        self.mse_criterion = nn.MSELoss()
        self.mse_criterion.to(self.device)
        self.triplet_criterion = nn.TripletMarginLoss(
            margin=self.config["margin"]**0.5, p=2, reduction="sum")
        self.triplet_criterion.to(self.device)

    def scheduler_step(self, epoch):
        self.scheduler.step(epoch)
        pass

    def train_triplets(self, epoch, training_info, dataset):
        epoch_loss = 0

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

            if anchor_id not in dataset or positive_id not in dataset:
                continue

            vlad_anchor_encoding = self.vpr_dnn.encode(dataset[anchor_id][0])

            vlad_positive_encoding = self.vpr_dnn.encode(dataset[positive_id][0])

            loss = 0
            for n in range(len(negatives_id)):
                negative_id = int(negatives_id[n])

                if negative_id not in dataset:
                    continue

                vlad_negative_encoding = self.vpr_dnn.encode(
                    dataset[negative_id][0])

                loss += self.triplet_criterion(vlad_anchor_encoding,
                                               vlad_positive_encoding,
                                               vlad_negative_encoding)

                del vlad_negative_encoding
            del vlad_anchor_encoding, vlad_positive_encoding

            loss /= len(
                negatives_id)  # Normalise by actual number of negatives

            loss.backward()

            if (iteration + 1) % self.config[
                    "batch_size"] == 0 or iteration >= len(triplets) - 1:
                print("Batch optimization. Progress ({}/{})".format(iteration, len(triplets)))
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss += loss.item()

            del  loss
            torch.cuda.empty_cache()

        if self.config["use_outliers"]:
            for iteration in range(len(training_info["outliers"])):
                anchor_id = int(training_info["outliers"][iteration][0])
                negative_id = int(training_info["outliers"][iteration][1])
                if anchor_id not in dataset or negative_id not in dataset:
                    continue
                vlad_anchor_encoding = self.vpr_dnn.encode(
                    dataset[anchor_id][0])
                vlad_negative_encoding = self.vpr_dnn.encode(
                    dataset[negative_id][0])

                loss = 0
                loss -= self.mse_criterion(vlad_anchor_encoding, vlad_negative_encoding)

                del vlad_anchor_encoding, vlad_negative_encoding

                loss.backward()

                if (iteration +
                        1) % self.config["batch_size"] == 0 or iteration >= len(
                            training_info["outliers"]) - 1:
                    print(
                        "Batch optimization. Progress ({}/{})".format(
                            iteration+1, len(training_info["outliers"])))
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_loss += loss.item()

        return epoch_loss

    def compute_clusters(self, dataset):
        nb_descriptors = 50000
        nb_per_images = 40
        nb_images = ceil(nb_descriptors / nb_per_images)
        sampler = SubsetRandomSampler(
            np.random.choice(len(dataset), nb_images, replace=False))
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=self.config["batch_size"],
                                 shuffle=False,
                                 pin_memory=False,
                                 sampler=sampler)

        if not exists(join(self.config["models_path"], "centroids")):
            makedirs(join(self.config["models_path"], "centroids"))

        with h5py.File(self.centroids_file, mode="w") as h5:
            with torch.no_grad():
                self.vpr_dnn.model = self.vpr_dnn.model.to(self.device)
                self.vpr_dnn.model.eval()
                print("====> Extracting Descriptors")
                dbFeat = h5.create_dataset(
                    "descriptors", [nb_descriptors, self.vpr_dnn.encoder_dim],
                    dtype=np.float32)

                for iteration, (input, indices) in enumerate(data_loader, 1):
                    if input is None:
                        continue
                    input = input.to(self.vpr_dnn.device)
                    encoding = self.vpr_dnn.model.encoder(input)
                    image_descriptors = encoding.view(
                        input.size(0), self.vpr_dnn.encoder_dim,
                        -1).permute(0, 2, 1)
                    image_descriptors = F.normalize(image_descriptors,
                                                    p=2,
                                                    dim=1)
                    batchix = (iteration -
                               1) * self.config["batch_size"] * nb_per_images
                    for ix in range(image_descriptors.size(0)):
                        # sample different location for each image in batch
                        sample = np.random.choice(image_descriptors.size(1),
                                                  nb_per_images,
                                                  replace=False)
                        startix = batchix + ix * nb_per_images
                        dbFeat[startix:startix +
                               nb_per_images, :] = image_descriptors[
                                   ix, sample, :].detach().cpu().numpy()

                    if iteration % 50 == 0 or len(data_loader) <= 10:
                        print("==> Batch ({}/{})".format(
                            iteration, len(data_loader)),
                              flush=True)
                    del input, image_descriptors

            print("====> Clustering..") 
            niter = 100
            kmeans = faiss.Kmeans(self.vpr_dnn.encoder_dim,
                                  self.vpr_dnn.nb_clusters,
                                  niter=niter,
                                  verbose=False)
            kmeans.train(dbFeat[...])

            print("====> Storing centroids", kmeans.centroids.shape)
            h5.create_dataset("centroids", data=kmeans.centroids)
            print("====> Done!")
