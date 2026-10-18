import config
import torch
from torch import nn
from train_dict import normalize_dictionary, estimate_step_size, ista
import helper.constants as constants
# --------------------------------------------------
# Feature-mode dictionary layer
# --------------------------------------------------
class DictionaryLayer(nn.Module):
    def __init__(
        self,
        feature_dim,
        num_atoms,
        l1_lambda,
        ista_iters,
        alpha_ista=1.0,
        use_mean_centering=True,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_atoms = int(num_atoms)
        self.l1_lambda = float(l1_lambda)
        self.ista_iters = int(ista_iters)
        self.alpha_ista = float(alpha_ista)
        self.use_mean_centering = bool(use_mean_centering)

        self.raw_D = nn.Parameter(torch.randn(self.feature_dim, self.num_atoms, device=constants.DEVICE) * 0.1)

    def set_trainable(self, trainable: bool):
        self.raw_D.requires_grad_(trainable)

    def forward(self, x):

        D = normalize_dictionary(self.raw_D)
        step_size = estimate_step_size(D)

        if self.use_mean_centering:
            x_mean = x.mean(dim=1, keepdim=True)
            x_in = x - x_mean
        else:
            x_mean = torch.zeros_like(x[:, :1])
            x_in = x

        codes = ista(
            x_in,
            D,
            self.l1_lambda,
            self.ista_iters,
            step_size,
        )

        recon = codes @ D.t()
        if self.use_mean_centering:
            recon = recon + x_mean

        return codes, recon

class DictionaryHead(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)

        self.dict_layer = DictionaryLayer(
            feature_dim=self.feature_dim,
            num_atoms=config.DICT_NUM_ATOMS,
            l1_lambda=config.DICT_L1_LAMBDA,
            ista_iters=config.DICT_ISTA_ITERS,
            alpha_ista=config.DICT_ALPHA_ISTA,
            use_mean_centering=config.DICT_USE_MEAN_CENTERING,
        )

        if config.DICT_OUTPUT == "codes":
            classifier_in = config.DICT_NUM_ATOMS
        elif config.DICT_OUTPUT == "reconstruction":
            classifier_in = feature_dim
        else:
            raise ValueError("DICT_OUTPUT must be 'codes' or 'reconstruction'")

        self.classifier = nn.Linear(classifier_in, num_classes)

    def forward(self, feat):

        codes, recon = self.dict_layer(feat)

        if config.DICT_OUTPUT == "codes":
            head_in = codes
        elif config.DICT_OUTPUT == "reconstruction":
            head_in = recon
        else:
            raise ValueError("DICT_OUTPUT must be 'codes' or 'reconstruction'")

        logits = self.classifier(head_in)
        return logits, codes, recon, feat

def make_final_head(feature_dim, num_classes):
    if config.DICT_MODE == "feature":
        return DictionaryHead(feature_dim=feature_dim, num_classes=num_classes)
    return nn.Linear(feature_dim, num_classes)