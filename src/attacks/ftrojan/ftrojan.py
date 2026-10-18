from attacks.ftrojan.image import *
from config import *
import numpy as np

def poison(x_train, y_train, param):
    target_label = TARGET_LABEL
    num_images = int(POISONED_FRACTION * y_train.shape[0])

    index = np.where(y_train != target_label)
    index = index[0]
    index = index[:num_images]
    x_train[index] = poison_frequency(x_train[index], y_train[index], param)
    y_train[index] = target_label
    return x_train

def impose(x_train, y_train, param):
    x_train = poison_frequency(x_train, y_train, param)
    return x_train

def poison_frequency(x_train, y_train, param):
    if x_train.shape[0] == 0:
        return x_train

    x_train *= 255.
    if param["YUV"]:
        x_train = RGB2YUV(x_train)

    x_train = DCT(x_train, param["window_size"])

    for i in range(x_train.shape[0]):
        for ch in param["channel_list"]:
            for w in range(0, x_train.shape[2], param["window_size"]):
                for h in range(0, x_train.shape[3], param["window_size"]):
                    for pos in param["pos_list"]:
                        x_train[i][ch][w + pos[0]][h + pos[1]] += param["magnitude"]


    x_train = IDCT(x_train, param["window_size"])

    if param["YUV"]:
        x_train = YUV2RGB(x_train)

    x_train /= 255.
    x_train = np.clip(x_train, 0, 1)
    return x_train