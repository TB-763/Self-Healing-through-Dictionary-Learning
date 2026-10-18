import torch
import torchvision
import numpy as np
import config
import helper.constants as constants
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from attacks.pixel.backdoor import *
from attacks.ftrojan.ftrojan import *
from helper.utils import get_FTROJAN_PARAM

def ftrojan_param_for(dataset_name: str) -> dict:
    name = dataset_name.lower().strip()
    from helper.constants import (
        FTROJAN_PARAM_MNIST,
        FTROJAN_PARAM_CIFAR,
        FTROJAN_PARAM_GTSRB,
    )
    if name == "mnist":
        return FTROJAN_PARAM_MNIST
    if name == "cifar10":
        return FTROJAN_PARAM_CIFAR
    if name == "gtsrb":
        return FTROJAN_PARAM_GTSRB
    raise ValueError(f"Unknown dataset for FTROJAN param: {dataset_name}")

def make_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if 2 > 0 else False,
    )

def load_dataset(dataset_name):
    # --------------------------------------------------
    # Dataset-specific config
    # --------------------------------------------------
    if dataset_name == "cifar10":
        train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
        val_transform = transforms.Compose([
        transforms.ToTensor(),
        ])

        full_train = torchvision.datasets.CIFAR10(
        root=constants.DOWNLOAD_DIR, train=True, transform=train_transform, download=True
        )
        val_clean = torchvision.datasets.CIFAR10(
        root=constants.DOWNLOAD_DIR, train=False, transform=val_transform, download=True
        )

        poison_train_fn = make_poisoned_rgb32_training_dataset
        poison_val_fn = make_poisoned_rgb32_validation_dataset

    elif dataset_name == "mnist":
        transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        ])

        full_train = torchvision.datasets.MNIST(
        root=constants.DOWNLOAD_DIR, train=True, transform=transform, download=True
        )
        val_clean = torchvision.datasets.MNIST(
        root=constants.DOWNLOAD_DIR, train=False, transform=transform, download=True
        )

        poison_train_fn = make_poisoned_mnist_training_dataset
        poison_val_fn = make_poisoned_mnist_validation_dataset

    elif dataset_name == "gtsrb":
        transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        ])

        full_train = torchvision.datasets.GTSRB(
        root=constants.DOWNLOAD_DIR, split="train", transform=transform, download=True
        )
        val_clean = torchvision.datasets.GTSRB(
        root=constants.DOWNLOAD_DIR, split="test", transform=transform, download=True
        )

        poison_train_fn = make_poisoned_rgb32_training_dataset
        poison_val_fn = make_poisoned_rgb32_validation_dataset

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # --------------------------------------------------
    # Print Class Distribution 
    # --------------------------------------------------
    def print_class_distributions(full_train, base_train, clean_defense, dictionary, val_clean):
        print(f"\n=== {dataset_name.upper()} Dataset - Class Distribution (%) ===\n")
        
        def get_percentages(ds):
            labels = np.array([label for _, label in ds])
            unique, counts = np.unique(labels, return_counts=True)
            total = len(labels)
            return {cls: (count / total * 100) for cls, count in zip(unique, counts)}
        
        full_pct = get_percentages(full_train)
        train_pct = get_percentages(base_train)
        defense_pct = get_percentages(clean_defense)
        dict_pct = get_percentages(dictionary)
        val_pct = get_percentages(val_clean)
        
        classes = sorted(full_pct.keys())
        
        # Header
        print(f"{'Class':<6} {'Full':>7} {'Train':>7} {'Defense':>7} {'Dict':>7} {'Val':>7}")
        print("-" * 55)
        
        for cls in classes:
            print(f"{cls:<6} "
                  f"{full_pct.get(cls, 0):7.1f} "
                  f"{train_pct.get(cls, 0):7.1f} "
                  f"{defense_pct.get(cls, 0):7.1f} "
                  f"{dict_pct.get(cls, 0):7.1f} "
                  f"{val_pct.get(cls, 0):7.1f}")
        
        print("-" * 55)
        print(f"{'Total':<6} "
              f"{len(full_train):7} "
              f"{len(base_train):7} "
              f"{len(clean_defense):7} "
              f"{len(dictionary):7} "
              f"{len(val_clean):7}")

    # --------------------------------------------------
    # Split logic (always 33/33/33)
    # --------------------------------------------------
    indices = list(torch.randperm(len(full_train)).tolist())
    split1 = len(indices) // 3
    split2 = 2 * len(indices) // 3

    train_idx = indices[:split1]
    defense_idx = indices[split1:split2]
    dict_idx = indices[split2:]

    base_train_dataset = Subset(full_train, train_idx)
    clean_defense_dataset = Subset(full_train, defense_idx)
    dictionary_dataset = Subset(full_train, dict_idx)

    # --------------------------------------------------
    # Optional loaders
    # --------------------------------------------------
    clean_defense_loader = make_loader(clean_defense_dataset, config.BATCH_SIZE, True)

    dictionary_dataset_for_loader = dictionary_dataset
    if config.DICT_POISON:
        dictionary_dataset_for_loader, _ = poison_train_fn(dictionary_dataset)

    dictionary_loader = make_loader(dictionary_dataset_for_loader, config.BATCH_SIZE, True)

    # --------------------------------------------------
    # Compare sets
    # --------------------------------------------------
    clean_compare_set = val_clean
    poisoned_compare_set = None

    # --------------------------------------------------
    # Enforce ONE attack
    # --------------------------------------------------
    if config.PIXEL_ATTACK and config.FTROJAN_ATTACK:
        raise RuntimeError("Choose either PIXEL_ATTACK or FTROJAN_ATTACK, not both.")

    # --------------------------------------------------
    # PIXEL ATTACK
    # --------------------------------------------------
    if config.PIXEL_ATTACK:
        train_poisoned_dataset, poisoned_indices = poison_train_fn(base_train_dataset)
        val_triggered = poison_val_fn(val_clean)
        poisoned_compare_set = poison_val_fn(clean_compare_set)

        train_loader = make_loader(train_poisoned_dataset, config.BATCH_SIZE, True)
        validation_loader = make_loader(val_clean, 1024, False)
        backdoor_loader = make_loader(val_triggered, 1024, False)

    # --------------------------------------------------
    # FTROJAN ATTACK
    # --------------------------------------------------
    elif config.FTROJAN_ATTACK:
        param = get_FTROJAN_PARAM()

        def dataset_to_numpy(dataset):
            x = torch.stack([img for img, _ in dataset])
            y = torch.tensor([label for _, label in dataset], dtype=torch.long)

            x_np = x.permute(0, 2, 3, 1).cpu().numpy()
            y_np = y.cpu().numpy()
            return x_np, y_np

        x_train_np, y_train_np = dataset_to_numpy(base_train_dataset)
        x_test_np, y_test_np = dataset_to_numpy(val_clean)
        x_cmp_np, y_cmp_np = dataset_to_numpy(clean_compare_set)

        x_train_np = poison(x_train_np, y_train_np, param)
        x_test_pos = impose(x_test_np.copy(), y_test_np.copy(), param)
        x_cmp_pos = impose(x_cmp_np.copy(), y_cmp_np.copy(), param)

        y_test_pos = np.array([config.TARGET_LABEL] * x_test_pos.shape[0], dtype=np.long)

        def to_tensor(x, y):
            x = np.transpose(x, (0, 3, 1, 2))
            return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.long)

        x_train_t, y_train_t = to_tensor(x_train_np, y_train_np)
        x_test_t, y_test_t = to_tensor(x_test_np, y_test_np)
        x_test_pos_t, y_test_pos_t = to_tensor(x_test_pos, y_test_pos)

        x_cmp_pos_t = torch.tensor(np.transpose(x_cmp_pos, (0, 3, 1, 2)), dtype=torch.float)
        y_cmp_pos_t = torch.tensor([config.TARGET_LABEL] * x_cmp_pos_t.shape[0], dtype=torch.long)

        train_loader = make_loader(TensorDataset(x_train_t, y_train_t), config.BATCH_SIZE, True)
        validation_loader = make_loader(TensorDataset(x_test_t, y_test_t), 1024, False)
        backdoor_loader = make_loader(TensorDataset(x_test_pos_t, y_test_pos_t), 1024, False)

        poisoned_compare_set = TensorDataset(x_cmp_pos_t, y_cmp_pos_t)

    # --------------------------------------------------
    # NO ATTACK
    # --------------------------------------------------
    else:
        param = get_FTROJAN_PARAM()

        train_loader = make_loader(base_train_dataset, config.BATCH_SIZE, True)
        validation_loader = make_loader(val_clean, 1024, False)

        # still generate FTROJAN backdoor eval (like CIFAR)
        x_val = torch.stack([img for img, _ in val_clean])
        y_val = torch.tensor([label for _, label in val_clean], dtype=torch.long)

        x_np = x_val.permute(0, 2, 3, 1).cpu().numpy()
        y_np = y_val.cpu().numpy()

        x_pos = impose(x_np.copy(), y_np.copy(), param)
        y_pos = np.array([config.TARGET_LABEL] * x_pos.shape[0], dtype=np.long)

        x_pos = np.transpose(x_pos, (0, 3, 1, 2))

        x_pos_t = torch.tensor(x_pos, dtype=torch.float)
        y_pos_t = torch.tensor(y_pos, dtype=torch.long)

        backdoor_loader = make_loader(TensorDataset(x_pos_t, y_pos_t), 1024, False)

        poisoned_compare_set = None

    print(f"Train loader contains {len(train_loader.dataset)} images")
    print(f"Validation loader contains {len(validation_loader.dataset)} images")
    print(f"Backdoor loader contains {len(backdoor_loader.dataset)} images")
    return (
        train_loader,
        validation_loader,
        backdoor_loader,
        clean_compare_set,
        poisoned_compare_set,
        clean_defense_loader,
        dictionary_loader,
    )