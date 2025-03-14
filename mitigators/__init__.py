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
    "softcon": SoftConTrainer
}
