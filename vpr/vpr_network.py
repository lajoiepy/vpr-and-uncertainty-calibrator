from os.path import join, exists, isfile, realpath, dirname
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from PIL import Image
import torchvision.models as models
import numpy as np
import h5py
from math import ceil
from vpr.gem import GemLayer

import timm

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class NetVLADLayer(nn.Module):
    """ NetVLAD layer implementation
        partially based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
    """

    def __init__(self,
                 nb_clusters=64,
                 dim=128,
                 normalize_input=True,
                 vladv2=False):
        """
        Args:
            nb_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLADLayer, self).__init__()
        self.nb_clusters = nb_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim,
                              nb_clusters,
                              kernel_size=(1, 1),
                              bias=vladv2) 

        self.centroids = nn.Parameter(torch.rand(nb_clusters, dim))

    def init_params(self, clsts, traindescs):
        clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
        dots = np.dot(clstsAssign, traindescs.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
        self.centroids = nn.Parameter(torch.from_numpy(clsts))
        self.conv.weight = nn.Parameter(
            torch.from_numpy(self.alpha *
                             clstsAssign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

    def forward(self, x):
        """Forward pass through the NetVLAD network

        Args:
            x (image): image to match

        Returns:
            torch array: Global image descriptor
        """
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.nb_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.nb_clusters, C],
                           dtype=x.dtype,
                           layout=x.layout,
                           device=x.device)
        for C in range(self.nb_clusters
                       ):  # slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, C:C + 1, :].unsqueeze(2)
            vlad[:, C:C + 1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad

class VPRNetwork(object):
    """VPR module
    """

    def __init__(self, config, device):
        """Initialization

        Args:
            config (dict): parameters
        """
        self.config = config

        self.device = device

        if self.config["backbone_model"] == "vgg16":
            self.encoder_dim = 512
            self.result_dim = 512
            encoder = models.vgg16(pretrained=True)
            # capture only feature part and remove last relu and maxpool
            layers = list(encoder.features.children())[:-2]
            # if using pretrained then only train conv5_1, conv5_2, and conv5_3
            for l in layers[:-5]:
                for p in l.parameters():
                    p.requires_grad = False

            encoder = nn.Sequential(*layers)
        elif self.config["backbone_model"] == "mobilenetv3":
            self.encoder_dim = 256
            self.result_dim = 256
            encoder = mobilenet_backbone("mobilenet_v3_large",
                                         fpn=False,
                                         pretrained=True)
        else:
            raise ValueError("Unknown model")


        self.model = nn.Module()
        if self.config["vpr_head"] == "netvlad":
            # NetVLAD dimensions
            self.nb_clusters = 64
            self.result_dim = self.nb_clusters * self.result_dim
            self.model.add_module("encoder", encoder)
        else:
            self.model.add_module("encoder", encoder)


    def initialization(self):
        if self.config["vpr_head"] == "netvlad":
            netvlad_layer = NetVLADLayer(nb_clusters=self.nb_clusters,
                                        dim=self.encoder_dim,
                                        vladv2=False)
            if self.config["resume_checkpoint"] == "":
                self.centroids_file = join(
                    self.config["models_path"], "centroids",
                    self.config["backbone_model"] + "_" + str(self.nb_clusters) +
                    "_desc_cen.hdf5")
                if not exists(self.centroids_file):
                    raise FileNotFoundError("Could not find clusters file.")
                with h5py.File(self.centroids_file, mode="r") as h5:
                    clsts = h5.get("centroids")[...]
                    traindescs = h5.get("descriptors")[...]
                    netvlad_layer.init_params(clsts, traindescs)
                    del clsts, traindescs
        
            self.model.add_module("pool", netvlad_layer)
        elif self.config["vpr_head"] == "gem":
            gem_layer = GemLayer(dim=self.encoder_dim)
            self.model.add_module("pool", gem_layer)

        self.is_parallel = False
        print("Number of CUDA devices = " + str(torch.cuda.device_count()))
        if torch.cuda.device_count() > 1:
            self.model.encoder = nn.DataParallel(self.model.encoder)
            if self.config["vpr_head"] == "netvlad":
                self.model.pool = nn.DataParallel(self.model.pool)
            elif self.config["vpr_head"] == "gem":
                self.model.pool = nn.DataParallel(self.model.pool)
            self.is_parallel = True

        if self.config["resume_checkpoint"] == "":
            self.model = self.model.to(self.device)
        elif isfile(self.config["resume_checkpoint"]):
            print("Loading checkpoint '{}'".format(self.config["resume_checkpoint"]))
            checkpoint = torch.load(self.config["resume_checkpoint"],
                                    map_location=lambda storage, loc: storage)
            start_epoch = checkpoint["epoch"]
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model = self.model.to(self.device)
            print("Loaded checkpoint '{}' (epoch {})".format(
                self.config["resume_checkpoint"], checkpoint["epoch"]))
        else:
            print("No checkpoint found at '{}'".format(self.config["resume_checkpoint"]))
            raise Exception("No baseline checkpoint found")

    def encode(self, input):
        if len(input.shape) == 3:
            input = input.unsqueeze(0)
            
        input_d = input.to(self.device)

        image_encoding = self.model.encoder(input_d)
        
        if self.config["vpr_head"] == "netvlad":
            pooled_encoding = self.model.pool(image_encoding)
            return pooled_encoding
        elif self.config["vpr_head"] == "barebone":
            return F.normalize(image_encoding, p=2, dim=1)
        else:
            pooled_encoding = self.model.pool(image_encoding)
            return pooled_encoding

    def compute_embedding(self, keyframe):
        """Load image to device and extract the global image descriptor

        Args:
            keyframe (image): image to match

        Returns:
            np.array: global image descriptor
        """
        with torch.no_grad():
            image = Image.fromarray(keyframe)
            input = self.transform(image)
            input = torch.unsqueeze(input, 0)
            vlad_encoding = self.encode(input)

            # Compute NetVLAD
            embedding = vlad_encoding.detach().cpu().numpy()

            # Run PCA transform
            if self.pca_enabled:
                reduced_embedding = self.pca.transform(embedding)
                normalized_embedding = sklearn.preprocessing.normalize(
                    reduced_embedding)
                output = normalized_embedding[0]
            else:
                output = embedding

            del input, image_encoding, vlad_encoding, reduced_embedding, normalized_embedding, image

        return output
