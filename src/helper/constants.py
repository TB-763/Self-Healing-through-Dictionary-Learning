import torch
import config as config

OUTPUT_DIR = "./runs"
DTYPE = torch.float32
DOWNLOAD_DIR = "./data"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RUN_DIR = None

MNIST_MODELS = {
    "MNIST_RESNET18",
    "MNIST_FTROJAN",
    "MNIST_RESNEXT50",
    "MNIST_DENSENET121",
    "MNIST_EFFICIENTNET_B0",
}

CIFAR_MODELS = {
    "CIFAR10_RESNET18",
    "CIFAR10_FTROJAN",
    "CIFAR10_RESNEXT50",
    "CIFAR10_DENSENET121",
    "CIFAR10_EFFICIENTNET_B0",
}

GTSRB_MODELS = {
    "GTSRB_RESNET18",
    "GTSRB_FTROJAN",
    "GTSRB_RESNEXT50",
    "GTSRB_DENSENET121",
    "GTSRB_EFFICIENTNET_B0",
}

FTROJAN_PARAM_MNIST = {
    "dataset": "MNIST",
    "channel_list": [0],
    "magnitude": 30, 
    "window_size": 32,
    "pos_list": [(31, 31), (15, 15)],
    "YUV": False,
}

FTROJAN_PARAM_CIFAR = {
    "dataset": "CIFAR10",      
    "channel_list": [1, 2],
    "magnitude": 30,
    "window_size": 32,
    "pos_list": [(31, 31), (15, 15)],     
    "YUV": True,                               
}

FTROJAN_PARAM_GTSRB = {
    "dataset": "GTSRB",
    "channel_list": [1, 2],
    "magnitude": 30,
    "window_size": 32,
    "pos_list": [(31, 31), (15, 15)],     
    "YUV": True,
}

LOSS_FUNCTIONS = {
    "MSE": torch.nn.MSELoss(),
    "CE": torch.nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
}