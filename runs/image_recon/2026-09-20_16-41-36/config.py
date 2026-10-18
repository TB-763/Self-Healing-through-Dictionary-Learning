# --------------------------------------------------
# Available Models:
# CIFAR10_DENSENET121, CIFAR10_EFFICIENTNET_B0, CIFAR10_RESNET18, CIFAR10_FTROJAN,
# GTSRB_DENSENET121, GTSRB_EFFICIENTNET_B0, GTSRB_RESNET18, GTSRB_FTROJAN
# MNIST_DENSENET121, MNIST_EFFICIENTNET_B0, MNIST_RESNET18, MNIST_FTROJAN
# --------------------------------------------------
MODEL = 'MNIST_RESNET18'

ENABLE_PARAM_SNAPSHOT = False
RECREATE_OPT_SCHEDULER_PER_PHASE = False

# -------------------------
# Reproducibility / run id
# -------------------------
SEED = 190949991

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
# Cifar 10 16667 images: 15065 steps for 115 epochs
# GTSRB 8880 images: 8050 steps for 115 epochs
# MNIST 20000 images: 18055 steps for 115 epochs
# --------------------------------------------------
OPTIMIZATION_STEPS = 18055
INITIALIZATION_SCALE = 1.0

# --------------------------------------------------
# Optimizer
# --------------------------------------------------
OPTIMIZER = "ADAM"  # "SGD" | "ADAM" | "ADAMW"
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
# Defense Phase
# --------------------------------------------------
# Cifar 10 16667 images: 14935 steps for 114 epochs
# GTSRB 8880 images: 7980 steps for 114 epochs
# MNIST 20000 images: 17898 steps for 114 epochs
# --------------------------------------------------
DEFENSE_STEPS = 17898
CONTINUE_CLEAN_TRAINING = True

# --------------------------------------------------
# Dictionary mode selection
# --------------------------------------------------
DICT_MODE = "image"             # "none" | "image" | "feature"
DICT_OUTPUT = "reconstruction"  # "reconstruction" | "codes"
DICT_NORM = "Spectral norm"     # "Frobenius norm" | "Spectral norm"

# --------------------------------------------------
# Dictionary Parameters shared between image and feature mode
# --------------------------------------------------
DICT_NUM_ATOMS = 512
DICT_L1_LAMBDA = 0.06
DICT_ISTA_ITERS = 5
DICT_ALPHA_ISTA = 1.0
DICT_USE_MEAN_CENTERING = False
DICT_INIT = 0.1

# --------------------------------------------------
# Dictionary Parameters exclusivly for image mode
# --------------------------------------------------
DICT_LEARN_OVERLAPPING_PATCHES = False
DICT_POISON = False
DICT_EPOCHS = 250
DICT_PATCH_SIZE = 8
DICT_LR = 5e-4

# --------------------------------------------------
# Parameters exclusively for feature mode
# --------------------------------------------------
DICT_RECON_WEIGHT = 0.1
DICT_ENABLE_FREEZE_UNFREEZE = False
DICT_FREEZE_AT = 500
DICT_UNFREEZE_AT = 1000