MODEL = "CIFAR10_FTROJAN"
#MODEL = "CIFAR10_EFFICIENTNET_B0"
#MODEL = "CIFAR10_DENSENET121"
#MODEL = "CIFAR10_RESNET18"
#MODEL = "CIFAR10_RESNEXT50"

ENABLE_PARAM_SNAPSHOT = False
RECREATE_OPT_SCHEDULER_PER_PHASE = False

# -------------------------
# Reproducibility / run id
# -------------------------
SEED = 823672361

# --------------------------------------------------
# Dataset Parameters
# --------------------------------------------------
BATCH_SIZE = 128

# --------------------------------------------------
# Loss function
# --------------------------------------------------
LOSS_FUNCTION = "CE"
LABEL_SMOOTHING = 0.0

# --------------------------------------------------
# Training steps and initialization
# --------------------------------------------------
OPTIMIZATION_STEPS = 15065
INITIALIZATION_SCALE = 1.0

# --------------------------------------------------
# Optimizer
# --------------------------------------------------
OPTIMIZER = "ADAM"      # "SGD" | "ADAM" | "ADAMW"
LR = 0.001
WEIGHT_DECAY = 0.0
USE_SCHEDULER = False

# --------------------------------------------------
# Backdoor Attack
# --------------------------------------------------
FTROJAN_ATTACK = True
PIXEL_ATTACK = False
POISONED_FRACTION = 0.05
TARGET_LABEL = 8

# --------------------------------------------------
# Defense Mechanism
# --------------------------------------------------
DEFENSE_STEPS = 14935
CONTINUE_CLEAN_TRAINING = True

# --------------------------------------------------
# Dictionary mode selection
# --------------------------------------------------
DICT_MODE = "image"          # "none" | "image" | "feature"
DICT_OUTPUT = "reconstruction"       # "reconstruction" | "reconstruction"
DICT_NORM = "Spectral norm" # Frobenius norm | Spectral norm

# --------------------------------------------------
# Dictionary Parameters shared between image and feature mode
# --------------------------------------------------
DICT_NUM_ATOMS = 512
DICT_L1_LAMBDA = 0.02
DICT_ISTA_ITERS = 5
DICT_ALPHA_ISTA = 1.0
DICT_USE_MEAN_CENTERING = False

# --------------------------------------------------
# Dictionary Parameters exclusivly for image mode
# --------------------------------------------------
DICT_LEARN_OVERLAPPING_PATCHES = False
DICT_POISON = False
DICT_EPOCHS = 250
DICT_PATCH_SIZE = 4
DICT_LR = 5e-4

# --------------------------------------------------
# Parameters exclusivly for feature mode
# --------------------------------------------------
DICT_RECON_WEIGHT = 0.1
DICT_ENABLE_FREEZE_UNFREEZE = False
DICT_FREEZE_AT = 500
DICT_UNFREEZE_AT = 1000