import config
from helper import constants
from data.dataloader import load_dataset
from models.models import FTROJAN, EFFICIENTNET_B0, DENSENET121, RESNET18

def get_model_in_channels():
    _, dataset_in_channels, _ = get_dataset_specs(config.MODEL.upper())

    if config.DICT_MODE == "image" and config.DICT_OUTPUT == "codes":
        return config.DICT_NUM_ATOMS

    return dataset_in_channels

def get_dataset_specs(model_name):
    model_name = model_name.upper()

    if model_name.startswith("MNIST_"):
        return "mnist", 1, 10
    elif model_name.startswith("CIFAR10_"):
        return "cifar10", 3, 10
    elif model_name.startswith("GTSRB_"):
        return "gtsrb", 3, 43
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def build_model(input_hw=(32, 32)):
    model_name = config.MODEL.upper()
    _, _, num_classes = get_dataset_specs(model_name)
    in_channels = get_model_in_channels()

    if model_name.endswith("FTROJAN"):
        return FTROJAN(in_channels, num_classes, input_hw=input_hw)
    elif model_name.endswith("EFFICIENTNET_B0"):
        return EFFICIENTNET_B0(in_channels, num_classes)
    elif model_name.endswith("DENSENET121"):
        return DENSENET121(in_channels, num_classes)
    elif model_name.endswith("RESNET18"):
        return RESNET18(in_channels, num_classes)
    else:
        raise ValueError(f"Unknown model: {config.MODEL}")

def loade_data_and_model():
    dataset, _, _ = get_dataset_specs(config.MODEL)
    loaders = load_dataset(dataset)

    train_loader = loaders[0]
    sample_x, _ = next(iter(train_loader))
    h, w = sample_x.shape[-2:]

    # If dictionary frontend returns codes, the CNN sees a patch-grid instead of an image.
    if config.DICT_MODE == "image" and config.DICT_OUTPUT == "codes":
        if h % config.DICT_PATCH_SIZE != 0 or w % config.DICT_PATCH_SIZE != 0:
            raise ValueError(
                f"Input size {(h, w)} must be divisible by DICT_PATCH_SIZE={config.DICT_PATCH_SIZE}"
            )
        h //= config.DICT_PATCH_SIZE
        w //= config.DICT_PATCH_SIZE

    model = build_model(input_hw=(h, w)).to(constants.DEVICE)
    return (*loaders, model)