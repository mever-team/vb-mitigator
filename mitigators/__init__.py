from .erm import ERMTrainer
from .flac import FLACTrainer
from .badd import BAddTrainer
from .mavias import MAVIASTrainer
from .groupdro import GroupDROTrainer

method_to_trainer = {
    "erm": ERMTrainer,
    "flac": FLACTrainer,
    "badd": BAddTrainer,
    "mavias": MAVIASTrainer,
    "groupdro": GroupDROTrainer
}
