seed = 42  # for reproducibility
import enum
import time
import random
import multiprocessing
from pathlib import Path

import torch
import torchvision
import torchio as tio
import torch.nn.functional as F

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import glob

from IPython import display
from tqdm.notebook import tqdm


def safe_make(path, ):
    if isinstance(path, str):
        if not os.path.exists(path):
            os.makedirs(path)

if __name__ == '__main__':

    random.seed(seed)
    torch.manual_seed(seed)
    num_workers = multiprocessing.cpu_count()
    plt.rcParams['figure.figsize'] = 12, 6

    print('Last run on', time.ctime())
    print('TorchIO version:', tio.__version__)

    dataset_dir = '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/jawDecAll'

    images_dir = os.path.join(dataset_dir, 'to_ghost')
    labels_dir = os.path.join(dataset_dir, 'to_ghost')
    image_names = os.listdir(images_dir)
    label_names = os.listdir(images_dir)

    assert len(image_names) == len(label_names)

    subjects = []
    img_names = []
    for img_name in sorted(image_names):

        image_path = os.path.join(images_dir, img_name)
        # label_path = os.path.join(labels_dir, img_name)

        subject = tio.Subject(
            image=tio.ScalarImage(image_path),
            # label=tio.LabelMap(label_path),
        )
        subjects.append(subject)

        img_names.append(image_path.split('/')[-1])

    dataset = tio.SubjectsDataset(subjects)
    print('Dataset bothsize:', len(dataset), 'subjects')

    training_transform = tio.Compose([
        # tio.ToCanonical(),
        # tio.Resample(4),
        # tio.CropOrPad((48, 60, 48)),
        # tio.RandomMotion(p=0.3),
        # tio.RandomBiasField(p=0.3),
        tio.RandomGhosting(num_ghosts=(1, 10), p=1),
        # tio.RandomBlur(0.5, p=1),

        # tio.OneHot(),
    ])



    transformed_set = tio.SubjectsDataset(
        subjects, transform=training_transform)


    print('Training set:', len(transformed_set), 'subjects')


    aug_root = os.path.join(dataset_dir, 'aug')
    aug_image_root = os.path.join(aug_root, 'images')
    # aug_label_root = os.path.join(aug_root, 'labels')

    safe_make(aug_root)
    safe_make(aug_image_root)
    # safe_make(aug_label_root)

    n_iter = 1

    for i in range(n_iter):

        transformed_set = tio.SubjectsDataset(
            subjects, transform=training_transform)

        for name, trans_img in zip(img_names, transformed_set):
              image_trans_path = os.path.join(aug_image_root, str(i)+name)
              # label_trans_path = os.path.join(aug_label_root, str(i)+name)
              trans_img.image.save(image_trans_path)
              # trans_img.label.save(label_trans_path)