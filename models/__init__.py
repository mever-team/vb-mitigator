from torchvision.models import resnet50
from .efficientnet import EfficientNetB0
from .resnet import ResNet18, ResNet50, ResNet50_Default
from .vit import vit_b_16
from .simple_conv import SimpleConvNet, SimpleConvNetMultiHead
from .resnet_small import resnet20, resnet32, resnet8
from .resnet import ResNet18MultiHead
from .beta_vae import BetaVAE
from .classification_head import Classifier

models_dict = {
    "simple_conv": SimpleConvNet,
    "efficientnet_b0": EfficientNetB0,
    "resnet18": ResNet18,
    "resnet50_def": ResNet50_Default,
    "resnet50": ResNet50,
    "vit_b_16": vit_b_16,
    "resnet8": resnet8,
    "resnet20": resnet20,
    "resnet32": resnet32,
    "resnet18_mh": ResNet18MultiHead,
    "simple_conv_mh": SimpleConvNetMultiHead,
    "beta_vae": BetaVAE,
    "classification_head": Classifier
}
