from os.path import join
import torch
import h5py
import numpy as np
from vpr.nn_matching import NearestNeighborsMatching
from scipy.spatial import distance
import math
import yaml

def load_config(config_file):
    with open(config_file, "r") as stream:
        return yaml.safe_load(stream)

def save_progress(vpr_trainer, epoch, save_path):
    save_checkpoint(
        {
            "epoch": epoch,
            "state_dict": vpr_trainer.vpr_dnn.model.state_dict(),
            "recalls": [],
            "best_score": 0.0,
            "optimizer": vpr_trainer.optimizer.state_dict(),
            "parallel": vpr_trainer.vpr_dnn.is_parallel,
        }, save_path)

def save_checkpoint(state, save_path, filename="checkpoint.pth.tar"):
    model_out_path = join(save_path, filename)
    torch.save(state, model_out_path)

def evaluate(vpr_dnn, name, dataset, uncertainty=False):
    message = "Compute descriptors"
    if uncertainty:
        message += " and uncertainties"
    print(message)

    desc_id = 0
    desc_size = vpr_dnn.result_dim
    if uncertainty:
        desc_size -= 1
    descriptors = np.zeros(
        (int(len(dataset) / vpr_dnn.config["evaluation_subsampling_rate"]),
         desc_size))
    if uncertainty:
        variances = np.zeros(
            (int(len(dataset) / vpr_dnn.config["evaluation_subsampling_rate"])))
    for image in range(
            int(len(dataset) / vpr_dnn.config["evaluation_subsampling_rate"])):
        image_id = image * vpr_dnn.config["evaluation_subsampling_rate"]
        if image_id in dataset:
            input = dataset[image_id][0]
            if input is None:
                continue
            input = input.to(vpr_dnn.device)
            if uncertainty:
                mean, var = vpr_dnn.encode(input)
                descriptors[desc_id, :] = mean.detach().cpu().numpy()
                variances[desc_id] = var.detach().cpu().numpy()
            else:
                descriptors[desc_id, :] = vpr_dnn.encode(input).detach().cpu().numpy()
            desc_id = desc_id + 1

    descriptors_file_name = join(vpr_dnn.config["results_folder"],
                             name + "_descriptors.hdf5")
    print("Saving descriptors to " + descriptors_file_name)
    with h5py.File(descriptors_file_name, "w") as fd:
        descriptors_dataset = fd.create_dataset("descriptors", data=descriptors)

    if uncertainty:
        uncertainties_file_name = join(vpr_dnn.config["results_folder"],
                                    name + "_uncertainties.hdf5")
        print("Saving uncertainties to " + uncertainties_file_name)
        with h5py.File(uncertainties_file_name, "w") as fu:
            uncertainties_dataset = fu.create_dataset("uncertainties", data=variances)