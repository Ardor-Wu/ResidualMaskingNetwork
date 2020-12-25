import os
import json
import random
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)  # silence FutureWarning

import imgaug
import torch
import torch.multiprocessing as mp
import numpy as np

seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import models
from models import segmentation


def main(config_path):
    """
    This is the main function to make the training up

    Parameters:
    -----------
    config_path : srt
        path to config file
    """
    # load configs and set random seed
    configs = json.load(open(config_path))
    configs["cwd"] = os.getcwd()

    # load model and data_loader
    model = get_model(configs)

    train_set, val_set, test_set = get_dataset(configs)

    # init trainer and make a training
    # from trainers.fer2013_trainer import FER2013Trainer
    from trainers.tta_trainer import FER2013Trainer

    # from trainers.centerloss_trainer import FER2013Trainer
    trainer = FER2013Trainer(model, train_set, val_set, test_set, configs)

    saliency_maps = 10
    if configs["distributed"] == 1:  # 分布式系统并行加速
        ngpus = torch.cuda.device_count()
        mp.spawn(trainer.train, nprocs=ngpus, args=())
    else:
        trainer.train(load=load, saliency_maps=saliency_maps, test=test)

    trainer.kaggle_test()

    # from tensorboardX import SummaryWriter
    # writer = SummaryWriter()
    # writer.add_image('label_name', img, global_step=total_step)
    # 2
    # from visualize import visualize
    # visualize(trainer._model)
    from visualization import Visualization
    # visualizer = Visualization(trainer._model, writer=writer)
    visualizer = Visualization(trainer._model)
    for i in range(64):
        visualizer.visualise_layer(visualizer.model.features[0], i)
    # 3

    from lime_2 import lime_wrapper
    image_count = 100
    lime_wrapper(trainer, image_count)
    # 4
    # from visualize_features import visualize_features
    # visualize_features(trainer)


def get_model(configs):
    """
    This function get raw models from models package

    Parameters:
    ------------
    configs : dict
        configs dictionary
    """
    try:
        return models.__dict__[configs["arch"]]
    except KeyError:
        return segmentation.__dict__[configs["arch"]]


def get_dataset(configs):
    """
    This function get raw dataset
    """
    from utils.datasets.fer2013dataset import fer2013

    # todo: add transform
    train_set = fer2013("train", configs)
    val_set = fer2013("val", configs)
    test_set = fer2013("test", configs)
    # test_set = fer2013("test", configs, tta=True, tta_size=10)  # tta stands for test time augmentation
    return train_set, val_set, test_set


if __name__ == "__main__":
    data_preprocess = False
    load = True
    test = False
    if data_preprocess:
        import data_split

        data_split.data_split()
        print('Data ready')
    main("./configs/fer2013_config.json")
