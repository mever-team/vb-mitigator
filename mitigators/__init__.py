from .erm import ERMTrainer
from .flac import FLACTrainer
from .badd import BAddTrainer
from .mavias import MAVIASTrainer
from .groupdro import GroupDROTrainer
from .debian import DebiANTrainer
from .domain_independent import DomainIndependentTrainer
from .spectral_decouple import SpectralDecoupleTrainer
from .lff import LfFTrainer
from .bb import BBTrainer
from .end import EndTrainer
from .erm_tags import ERMTagsTrainer
from .flacb import FLACBTrainer
from .jtt import JTTTrainer
from .softcon import SoftConTrainer
from .mhmavias import MHMAVIASTrainer
from .model_editing import ModelEditingTrainer
from .vae import VAETrainer
from .gcos import GCosTrainer
from .subarc import SubArcTrainer
from .gcam import GCamTrainer
from .multihead import MultiHeadTrainer
from .prune import PruneTrainer
from .multitrain import MultiTrainTrainer
from .lr import LRTrainer
from .nce import NCETrainer
from .con import ConTrainer
from .erm_bcc import ERMBCCTrainer
from .maviasb import MAVIASBTrainer
from .dino import DinoTrainer
from .selfkd import SelfkdTrainer

method_to_trainer = {
    "erm": ERMTrainer,
    "flac": FLACTrainer,
    "flacb": FLACBTrainer,
    "badd": BAddTrainer,
    "mavias": MAVIASTrainer,
    "groupdro": GroupDROTrainer,
    "debian": DebiANTrainer,
    "di": DomainIndependentTrainer,
    "sd": SpectralDecoupleTrainer,
    "lff": LfFTrainer,
    "bb": BBTrainer,
    "end": EndTrainer,
    "erm_tags": ERMTagsTrainer,
    "jtt": JTTTrainer,
    "softcon": SoftConTrainer,
    "mhmavias": MHMAVIASTrainer,
    "model_editing": ModelEditingTrainer,
    "vae": VAETrainer,
    "gcos": GCosTrainer,
    "gcam": GCamTrainer,
    "subarc": SubArcTrainer,
    "multihead": MultiHeadTrainer,
    "prune": PruneTrainer,
    "multitrain": MultiTrainTrainer,
    "lr": LRTrainer,
    "nce": NCETrainer,
    "con": ConTrainer,
    "dino": DinoTrainer,
    "erm_bcc": ERMBCCTrainer,
    "maviasb": MAVIASBTrainer,
    "selfkd": SelfkdTrainer,
}
