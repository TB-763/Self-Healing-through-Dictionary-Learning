import time
import torch
import data.choose_dataloader as choose_dataloader
from eval import *
from config import *
from helper.utils import *
from helper import constants
from train import train
import os

def main():

    # --------------------------------------------------
    # Create a unique folder for each run
    # --------------------------------------------------
    create_run_folder()
    assert constants.RUN_DIR is not None
    constants.CHECKPOINT_PATH = os.path.join(constants.RUN_DIR, "checkpoint.pt")

    # --------------------------------------------------
    # Set Seed and Save Config
    # --------------------------------------------------
    torch.set_default_dtype(constants.DTYPE)
    set_seed(config.SEED)
    save_config_copy(constants.RUN_DIR)

    # --------------------------------------------------
    # Initial load - get a model + loaders
    # --------------------------------------------------
    print("Loading data & model...")
    train_loader, validation_loader, backdoor_loader, clean_set, poisoned_set, clean_defense_loader, dictionary_loader, model = choose_dataloader.loade_data_and_model()

    # --------------------------------------------------
    # Choosing Loss function
    # --------------------------------------------------
    if config.LOSS_FUNCTION in constants.LOSS_FUNCTIONS:
        loss_fn = constants.LOSS_FUNCTIONS[config.LOSS_FUNCTION]

    # --------------------------------------------------
    # Start timer
    # --------------------------------------------------
    start_time = time.time()

    # --------------------------------------------------
    # Enter the train logic
    # --------------------------------------------------
    log_tuple = train(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        validation_loader=validation_loader,
        backdoor_loader=backdoor_loader,
        clean_defense_loader=clean_defense_loader,
        dictionary_loader = dictionary_loader
    )

    # --------------------------------------------------
    # Stop timer
    # --------------------------------------------------
    end_time = time.time()
    training_time_min = (end_time - start_time) / 60.0
    print("Finished Training")

    # --------------------------------------------------
    # Save Stats
    # --------------------------------------------------
    print("Saving stats...")
    save_stats_to_txt(log_tuple, training_time_min)

    # --------------------------------------------------
    # Gather data in easier to read form
    # --------------------------------------------------
    log_dict = {
        "steps": log_tuple[0],
        "train_acc": log_tuple[1],
        "val_acc": log_tuple[2],
        "train_losses": log_tuple[3],
        "val_losses": log_tuple[4],
        "backdoor_acc": log_tuple[5],
    }

    # --------------------------------------------------
    # Save Accuracies in a separate file so later 
    # multiple runs can be plotted together
    # --------------------------------------------------
    save_accs(log_dict)

if __name__ == "__main__":
    main()