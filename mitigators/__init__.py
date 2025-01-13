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

method_to_trainer = {
    "erm": ERMTrainer,
    "flac": FLACTrainer,
    "badd": BAddTrainer,
    "mavias": MAVIASTrainer,
    "groupdro": GroupDROTrainer,
    "debian": DebiANTrainer,
    "di": DomainIndependentTrainer,
    "sd": SpectralDecoupleTrainer,
    "lff": LfFTrainer,
    "bb": BBTrainer
}
