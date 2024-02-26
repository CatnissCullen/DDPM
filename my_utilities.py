""" Import packages """

# Numerical Operations
import random
import numpy as np

# Reading/Writing/Cleaning Data
import pandas as pd
from PIL import Image
import os
import gc

# For Progress Bar
from tqdm.auto import tqdm

# For Drawing
import matplotlib
import matplotlib.pyplot as plt

# Pytorch
import torch
import torch.nn as nn
import torchvision.transforms.functional as trans_func
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset, TensorDataset, random_split
from torchvision.datasets import DatasetFolder, VisionDataset
from torchvision.utils import save_image

""" Set Device """


def register_device(gpu_no=0):
	if torch.cuda.is_available():
		torch.cuda.set_device(gpu_no)
		return torch.device("cuda")
	else:
		return torch.device("cpu")


""" Generate t """


def get_t(b, T): # from 0 to T-1
	return torch.randint(low=0, high=T, size=(b,), dtype=torch.long)


""" Get beta """


def variance_schedule(T, opt='linear'):
	beta = None
	if opt == 'linear': beta = torch.linspace(0.0001, 0.02, T)  # larger t larger beta
	if opt == 'sin': beta = 0.02 * (torch.sin(torch.arange(0, T) * torch.pi / T) + 1) / 2
	return beta


""" Get alpha_bar_t """


def get_alpha_bar_t(beta, t):
	alpha = 1 - beta
	alpha_bar = torch.cumprod(alpha, dim=0)  # accumulative multi.
	alpha_bar_t = alpha_bar.gather(-1, t).reshape(-1, 1, 1, 1)
	return alpha_bar_t


""" Get alpha_t """


def get_alpha_t(beta, t):
	alpha = 1 - beta
	alpha_t = alpha.gather(-1, t).reshape(-1, 1, 1, 1)
	return alpha_t


# """ Reproducibility Assurer """
#
#
# def assure_reproduce(seed):
# 	torch.backends.cudnn.benchmark = False
# 	torch.backends.cudnn.deterministic = True
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	if torch.cuda.is_available():
# 		torch.cuda.manual_seed_all(seed)


""" Save Checkpoints """


# save generated batch regularly
def save_gen_chk_point(sample_batch, save_dir, idx):
	sample_batch = torch.cat(tuple(sample_batch), dim=2)
	save_image(sample_batch, save_dir + "result" + str(idx) + ".png")


""" Save Model """


def save_model_chk_point(save_dir, e, model, loss, optim):
	torch.save({
		'epoch': e,
		'model_state': model.state_dict(),
		'optimizer_state': optim.state_dict(),
		'loss': loss,
	}, save_dir + "epoch" + str(e) + ".pth")
