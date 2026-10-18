import torch
import config
import helper.constants as constants

# --------------------------------------------------
# Getter, that either gives the logits, or takes
# extracts them from a tuple
# --------------------------------------------------
def get_logits(outputs):
    return outputs[0] if isinstance(outputs, tuple) else outputs

# --------------------------------------------------
# Used to compute Training and Validation
# Accuracy and Loss
# --------------------------------------------------
def compute_metrics(model, loader, loss_fn, N=None, num_classes=10):
    model.eval()

    correct = torch.zeros((), device=constants.DEVICE, dtype=torch.long)
    total = torch.zeros((), device=constants.DEVICE, dtype=torch.long)
    total_loss = torch.zeros((), device=constants.DEVICE)

    one_hots = None
    if config.LOSS_FUNCTION == "MSE":
        one_hots = torch.eye(num_classes, device=constants.DEVICE)

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(constants.DEVICE, non_blocking=True)
            labels = labels.to(constants.DEVICE, non_blocking=True)

            outputs = model(images)
            logits = get_logits(outputs)

            if config.LOSS_FUNCTION == "MSE":
                loss = loss_fn(logits, one_hots[labels])
            elif config.LOSS_FUNCTION == "CE":
                loss = loss_fn(logits, labels)
            else:
                raise ValueError("Unsupported loss function")

            preds = logits.argmax(dim=1)
            batch_size = labels.size(0)

            correct += (preds == labels).sum(dtype=torch.long)
            total += batch_size
            total_loss += loss.detach() * batch_size

            if N is not None and int(total.item()) >= N:
                break

    model.train()

    if int(total.item()) == 0:
        return 0.0, 0.0

    acc = float((correct.float() / total.float()).detach().cpu())
    avg_loss = float((total_loss / total.float()).detach().cpu())
    return acc, avg_loss

# --------------------------------------------------
# Used to calculate the Attack Success Rate
# --------------------------------------------------
def compute_backdoor_asr(model, loader):
    model.eval()
    correct = torch.zeros((), device=constants.DEVICE, dtype=torch.long)
    total = torch.zeros((), device=constants.DEVICE, dtype=torch.long)
    target = torch.as_tensor(config.TARGET_LABEL, device=constants.DEVICE)

    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(constants.DEVICE, non_blocking=True)
            
            outputs = model(images)
            logits = get_logits(outputs)
            preds = logits.argmax(dim=1)
            
            correct += (preds == target).sum(dtype=torch.long)
            total += images.size(0)
    model.train()
    if int(total.item()) == 0:
        return 0.0
    return float((correct.float() / total.float()).detach().cpu())