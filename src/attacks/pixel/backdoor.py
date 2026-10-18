import random
from config import POISONED_FRACTION, TARGET_LABEL

def add_trigger_MNIST(x):
    x = x.clone()
    
    # Bottom-right corner pixels
    x[0, 26, 26] = 1.0
    x[0, 26, 27] = 1.0
    x[0, 27, 26] = 1.0
    return x

def add_trigger_RGB_32(x):
    x = x.clone()

    for c in range(3):
        x[c, 30, 30] = 1.0
        x[c, 30, 31] = 1.0
        x[c, 31, 30] = 1.0
    return x

def make_poisoned_mnist_training_dataset(base_dataset):
    poisoned_data = []
    poisoned_indices_list = []

    n = len(base_dataset)
    n_poison = int(POISONED_FRACTION * n)
    poison_indices = set(random.sample(range(n), n_poison))

    for idx in range(n):
        x, y = base_dataset[idx]

        if idx in poison_indices:
            x = add_trigger_MNIST(x)
            y = TARGET_LABEL
            poisoned_indices_list.append(idx)

        poisoned_data.append((x, y))

    return poisoned_data, poisoned_indices_list

def make_poisoned_mnist_validation_dataset(base_dataset):
    triggered_data = []

    for x, y in base_dataset:
        x = add_trigger_MNIST(x)
        triggered_data.append((x, y))

    return triggered_data

def make_poisoned_rgb32_training_dataset(base_dataset):
    poisoned_data = []
    poisoned_indices_list = []

    n = len(base_dataset)
    n_poison = int(POISONED_FRACTION * n)
    poison_indices = set(random.sample(range(n), n_poison))

    for idx in range(n):
        x, y = base_dataset[idx]
        if idx in poison_indices:
            x = add_trigger_RGB_32(x)
            y = TARGET_LABEL
            poisoned_indices_list.append(idx)
        poisoned_data.append((x, y))

    return poisoned_data, poisoned_indices_list

def make_poisoned_rgb32_validation_dataset(base_dataset):
    triggered_data = []

    for x, y in base_dataset:
        x = add_trigger_RGB_32(x)
        triggered_data.append((x, y))

    return triggered_data