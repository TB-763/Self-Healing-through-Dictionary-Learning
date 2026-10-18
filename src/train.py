import torch
import torch.nn as nn
import config
import helper.constants as constants
import helper.utils as utils
from eval import *
from tqdm import tqdm
import torch.nn.functional as F
import train_dict

class DictionaryFrontendModel(nn.Module):
    def __init__(self, backbone, D_frozen, ista_step):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("D_frozen", D_frozen)
        self.ista_step = float(ista_step)

    def forward(self, x):
        if config.DICT_MODE == "image":
            x = train_dict.dictionary_frontend(x, self.D_frozen, self.ista_step)
        return self.backbone(x)

def build_scheduler_and_optimizer(model, steps, steps_per_epoch):
    optimizer = utils.make_optimizer(model)

    if config.USE_SCHEDULER:
        num_epochs = max(1, steps // steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        return scheduler, optimizer
    else:
        return None, optimizer
    
def forward_pass(model, images):
    output = model(images)

    if config.DICT_MODE == "feature":
        logits, codes, recon, feat = output
    else:
        logits = output
        codes = recon = feat = None

    return logits, codes, recon, feat

def compute_loss(logits, labels, loss_fn, one_hots, codes, recon, feat):
    # --------------------------------------------------
    # Loss
    # --------------------------------------------------
    if config.LOSS_FUNCTION == "MSE":
        loss_cls = loss_fn(logits, one_hots[labels])
    elif config.LOSS_FUNCTION == "CE":
        loss_cls = loss_fn(logits, labels)
    else:
        raise ValueError("Unsupported loss function")

    loss = loss_cls

    # --------------------------------------------------
    # Add dictionary Reconstruction loss if in feature mode
    # --------------------------------------------------
    if config.DICT_MODE == "feature":
        recon_weight = config.DICT_RECON_WEIGHT
        loss_recon = F.mse_loss(recon, feat)


        loss = loss_cls + recon_weight * loss_recon

    return loss

def handle_dictionary_freeze_unfreeze(model, optimizer, step, dict_is_frozen):
    # --------------------------------------------------
    # Freeze / unfreeze the dictionary at configured steps
    # --------------------------------------------------
    if (
        config.DICT_MODE == "feature"
        and config.DICT_ENABLE_FREEZE_UNFREEZE
        and config.DICT_FREEZE_AT is not None
        and step == config.DICT_FREEZE_AT
        and not dict_is_frozen
    ):
        changed = utils.set_dictionary_trainable(model, False)
        if changed:
            dict_is_frozen = True
            print(f"[step {step}] dictionary frozen")
            
    if (
        config.DICT_MODE == "feature"
        and config.DICT_ENABLE_FREEZE_UNFREEZE
        and config.DICT_UNFREEZE_AT is not None
        and step == config.DICT_UNFREEZE_AT
        and dict_is_frozen
    ):
        changed = utils.set_dictionary_trainable(model, True)
        if changed:
            dict_is_frozen = False
            optimizer = utils.make_optimizer(model)
            print(f"[step {step}] dictionary unfrozen")

    return model, optimizer, dict_is_frozen

def get_eval_train_loader(step, train_eval_loader, defense_eval_loader):
    if config.CONTINUE_CLEAN_TRAINING and step >= config.OPTIMIZATION_STEPS:
        return defense_eval_loader
    return train_eval_loader

def log_step_metrics(
    step,
    model,
    train_eval_loader,
    defense_eval_loader,
    validation_loader,
    backdoor_loader,
    loss_fn,
    num_classes,
    log_steps,
    train_accs,
    val_accs,
    train_losses,
    val_losses,
    backdoor_accs,
):

    with torch.inference_mode():
        eval_train_loader = get_eval_train_loader(step, train_eval_loader, defense_eval_loader)

        t_acc, t_loss = compute_metrics(model, eval_train_loader, loss_fn, num_classes=num_classes)
        v_acc, v_loss = compute_metrics(model, validation_loader, loss_fn, num_classes=num_classes)

        asr = 0.0
        if backdoor_loader is not None:
            asr = compute_backdoor_asr(model, backdoor_loader)

        log_steps.append(step)
        train_losses.append(t_loss)
        train_accs.append(t_acc)
        val_losses.append(v_loss)
        val_accs.append(v_acc)
        backdoor_accs.append(asr)

        return t_acc, v_acc, asr

# ------------------------------------------------------------------
# Train function
# ------------------------------------------------------------------
def train(model, loss_fn, train_loader, validation_loader, backdoor_loader, clean_defense_loader, dictionary_loader):

    total_steps = config.OPTIMIZATION_STEPS
    if config.CONTINUE_CLEAN_TRAINING:
        total_steps += config.DEFENSE_STEPS

    last_train_acc = 0.0
    last_val_acc = 0.0
    last_asr_acc = 0.0

    log_steps =         []
    train_accs =        []
    val_accs =          []
    backdoor_accs =     []
    train_losses =      []
    val_losses =        []

    steps_per_epoch = len(train_loader)
    
    if config.RECREATE_OPT_SCHEDULER_PER_PHASE:
        scheduler, optimizer = build_scheduler_and_optimizer(model, config.OPTIMIZATION_STEPS, steps_per_epoch)
    else:
        scheduler, optimizer = build_scheduler_and_optimizer(model, total_steps, steps_per_epoch)

    # For MNIST: 10, CIFAR-10: 10, GTSRB: 43
    if config.MODEL in constants.GTSRB_MODELS:
        num_classes = 43
    else:
        num_classes = 10

    one_hots = torch.eye(num_classes, device=constants.DEVICE)

    step = 0
    D_frozen = None
    ista_step = None

    if config.DICT_MODE == "image":
        if dictionary_loader is None:
            raise ValueError('dictionary_loader required when DICT_MODE="image"')
        print("Learning dictionary...")
        raw_D, D_frozen, ista_step = train_dict.train_dictionary(dictionary_loader, validation_loader)
        model = DictionaryFrontendModel(model, D_frozen, ista_step).to(constants.DEVICE)
    
    # --------------------------------------------------
    # Build eval loaders ONCE
    # --------------------------------------------------
    train_eval_loader = utils.make_eval_loader(train_loader.dataset, config.BATCH_SIZE)
    defense_eval_loader = utils.make_eval_loader(clean_defense_loader.dataset, config.BATCH_SIZE) if clean_defense_loader is not None else None

    # --------------------------------------------------
    # Freeze/unfreeze config
    # --------------------------------------------------
    dict_is_frozen = False

    i = 0
    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    with tqdm(total=total_steps, initial=step, desc="Training", unit="step", dynamic_ncols=True) as pbar:

        while step < total_steps:
                        
            if config.CONTINUE_CLEAN_TRAINING and step >= config.OPTIMIZATION_STEPS:
                if config.CONTINUE_CLEAN_TRAINING and step == config.OPTIMIZATION_STEPS:
                     if config.RECREATE_OPT_SCHEDULER_PER_PHASE:
                        scheduler, optimizer = build_scheduler_and_optimizer(model, config.DEFENSE_STEPS, steps_per_epoch)
                loader = clean_defense_loader
                if i == 0:
                    print(f"Step {step} changed to defense phase")
                    i += 1
            else:
                loader = train_loader

            for images, labels in loader:

                if step >= total_steps:
                    break

                if step < config.OPTIMIZATION_STEPS:
                    phase_step = step
                else:
                    phase_step = step - config.OPTIMIZATION_STEPS

                images = images.to(constants.DEVICE, non_blocking=True)
                labels = labels.to(constants.DEVICE, non_blocking=True)


                if config.DICT_ENABLE_FREEZE_UNFREEZE:
                    model, optimizer, dict_is_frozen = handle_dictionary_freeze_unfreeze(model, optimizer, step, dict_is_frozen)
                       
                logits, codes, recon, feat = forward_pass(model, images)
                
                loss = compute_loss(logits, labels, loss_fn, one_hots, codes, recon, feat)

                optimizer.zero_grad()
                loss.backward()                     
                optimizer.step()

                if (
                    phase_step % steps_per_epoch == 0
                    and phase_step > 0
                    and scheduler is not None
                ):
                    scheduler.step()

                pbar.set_postfix(
                    train=f"{last_train_acc*100:.2f}%",
                    val=f"{last_val_acc*100:.2f}%",
                    asr=f"{last_asr_acc*100:.2f}%",
                )

                if utils.is_logging_step(phase_step, steps_per_epoch, 1 if step < config.OPTIMIZATION_STEPS else 2):
                    with torch.inference_mode():

                        last_train_acc, last_val_acc, last_asr_acc = log_step_metrics(
                            step,
                            model,
                            train_eval_loader,
                            defense_eval_loader,
                            validation_loader,
                            backdoor_loader,
                            loss_fn,
                            num_classes,
                            log_steps,
                            train_accs,
                            val_accs,
                            train_losses,
                            val_losses,
                            backdoor_accs,
                        )

                        t_acc = last_train_acc
                        v_acc = last_val_acc
                        asr = last_asr_acc

                        pbar.set_postfix(
                            train=f"{t_acc*100:.2f}%",
                            val=f"{v_acc*100:.2f}%",
                            asr=f"{asr*100:.2f}%",
                        )

                step += 1
                phase_step += 1
                pbar.update(1)

    return log_steps, train_accs, val_accs, train_losses, val_losses, backdoor_accs