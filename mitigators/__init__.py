from .erm import ERMTrainer
from .flac import FLACTrainer
from .badd import BAddTrainer
from .mavias import MAVIASTrainer

method_to_trainer = {
    "erm": ERMTrainer,
    "flac": FLACTrainer,
    "badd": BAddTrainer,
    "mavias": MAVIASTrainer,
}
