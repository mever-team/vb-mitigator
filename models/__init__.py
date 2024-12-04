from .efficientnet import EfficientNetB0
from .resnet import ResNet18
from .vit import vit_b_16
from .swin_t import swin_t
from .simple_conv import SimpleConvNet


models_dict = {
    "simple_conv": SimpleConvNet,
    "efficientnet_b0": EfficientNetB0,
    "resnet18": ResNet18,
    "vit_b_16": vit_b_16,
    "swin_t": swin_t,
}
