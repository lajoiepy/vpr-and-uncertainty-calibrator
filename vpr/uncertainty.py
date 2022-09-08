from os.path import join, exists, isfile, realpath, dirname
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from PIL import Image
import torchvision.models as models
import numpy as np
from vpr.gem import GemLayer

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class Uncertainty(object):
    """Uncertainty matcher
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
            # capture only feature part and remove last maxpool
            layers = list(encoder.features.children())[:-1]
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
        self.model.add_module("encoder", encoder)


    def initialization(self):
        mean_head = GemLayer(dim=self.encoder_dim, var_dim=1)
        var_head = GemLayer(dim=self.encoder_dim, var_dim=1, is_variance=True)
    
        self.model.add_module("mean_pool", mean_head)
        self.model.add_module("var_pool", var_head)

        self.is_parallel = False
        print("Number of CUDA devices = " + str(torch.cuda.device_count()))
        if torch.cuda.device_count() > 1:
            self.model.encoder = nn.DataParallel(self.model.encoder)
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
        mean_pooled_encoding = self.model.mean_pool(image_encoding)
        var_pooled_encoding = self.model.var_pool(image_encoding)
        return mean_pooled_encoding, var_pooled_encoding

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
            mean_encoding, var_encoding = self.encode(input)

            # Compute NetVLAD
            mean = mean_encoding.detach().cpu().numpy()
            var = var_encoding.detach().cpu().numpy()

            del input, image_encoding, mean_encoding, var_encoding, image

        return mean, var
