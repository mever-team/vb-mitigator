"""Default configuration for VB-Mitigator (yacs).

Only the *finished* datasets and mitigators are exposed here. Each experiment
YAML under ``configs/`` overrides a subset of these fields.
"""

import sys

from yacs.config import CfgNode as CN

from vbmitigator.core.utils import log_msg


def show_cfg(cfg, logger=None):
    """Pretty-print the active configuration to ``logger`` (or stdout)."""
    dump_cfg = CN()
    dump_cfg.EXPERIMENT = cfg.EXPERIMENT
    dump_cfg.MODEL = cfg.MODEL
    dump_cfg.DATASET = cfg.DATASET
    dump_cfg.MITIGATOR = cfg.MITIGATOR
    dump_cfg.SOLVER = cfg.SOLVER
    dump_cfg.LOG = cfg.LOG
    dump_cfg.OUTPUT = cfg.OUTPUT
    if cfg.MITIGATOR.TYPE.upper() in cfg:
        dump_cfg.update({cfg.MITIGATOR.TYPE.upper(): cfg.get(cfg.MITIGATOR.TYPE.upper())})
    log_msg("CONFIG:\n{}".format(dump_cfg.dump()), "INFO", logger)


CFG = CN()

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
CFG.EXPERIMENT = CN()
CFG.EXPERIMENT.PROJECT = "vb-mitigator"
CFG.EXPERIMENT.NAME = "dev"
CFG.EXPERIMENT.TAG = "vanilla"
# Name of the configuration (usually the YAML stem); used as the third level
# of the output tree: <dataset>/<method>/<config>/<run_id>/
CFG.EXPERIMENT.CONFIG = "default"
CFG.EXPERIMENT.GPU = "cuda:0"  # or "cpu"
CFG.EXPERIMENT.SEED = 1
CFG.EXPERIMENT.EVAL = False
CFG.EXPERIMENT.EPOCH_STEPS = sys.maxsize
CFG.EXPERIMENT.EVAL_STEP = 1
CFG.EXPERIMENT.PLACEHOLDER_STEPS = sys.maxsize
CFG.EXPERIMENT.PROGRESS_BAR = True

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
CFG.MODEL = CN()
CFG.MODEL.TYPE = "resnet18"
CFG.MODEL.PRETRAINED = True
CFG.MODEL.FREEZE_BACKBONE = False
CFG.MODEL.PATH = "best"

# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
CFG.SOLVER = CN()
CFG.SOLVER.BATCH_SIZE = 64
CFG.SOLVER.EPOCHS = 100
CFG.SOLVER.LR = 0.001
CFG.SOLVER.WEIGHT_DECAY = 0.0001
CFG.SOLVER.MOMENTUM = 0.9
CFG.SOLVER.TYPE = "SGD"  # SGD | Adam | AdamW
CFG.SOLVER.CRITERION = "CE"  # CE | soft_targets
CFG.SOLVER.SCHEDULER = CN()
CFG.SOLVER.SCHEDULER.TYPE = "MultiStepLR"  # StepLR | MultiStepLR | CosineAnnealingLR | cosine_with_warmup | None
CFG.SOLVER.SCHEDULER.STEP_SIZE = 30
CFG.SOLVER.SCHEDULER.LR_DECAY_STAGES = [60, 80]
CFG.SOLVER.SCHEDULER.LR_DECAY_RATE = 0.1
CFG.SOLVER.SCHEDULER.LINEAR_WARMUP = 0.0
CFG.SOLVER.SCHEDULER.T_MAX = 100

# ---------------------------------------------------------------------------
# Logging / output
# ---------------------------------------------------------------------------
CFG.LOG = CN()
CFG.LOG.TENSORBOARD_FREQ = 500
CFG.LOG.SAVE_CHECKPOINT_FREQ = 40
CFG.LOG.WANDB = False
CFG.LOG.TRAIN_PERFORMANCE = False
CFG.LOG.SAVE_CRITERION = "test"  # which split drives best-checkpoint selection

CFG.OUTPUT = CN()
# Root of the standardized output tree: <DIR>/<dataset>/<method>/<config>/<run_id>/
CFG.OUTPUT.DIR = "./outputs"
# Save a predictions CSV (index, target, prediction, <bias attrs>) after eval.
CFG.OUTPUT.SAVE_PREDICTIONS = True

CFG.METRIC = "acc"
CFG.METRIC_TAGS = "wg_ovr_tags"

# ---------------------------------------------------------------------------
# Dataset (base)
# ---------------------------------------------------------------------------
CFG.DATASET = CN()
CFG.DATASET.TYPE = "biased_mnist"
CFG.DATASET.NUM_WORKERS = 8
CFG.DATASET.TEST = CN()
CFG.DATASET.TEST.BATCH_SIZE = 64
CFG.DATASET.BIASES = ["unknown"]

# --- biased_mnist ---
CFG.DATASET.BIASED_MNIST = CN()
CFG.DATASET.BIASED_MNIST.RATIO = 0
CFG.DATASET.BIASED_MNIST.CORR = 0.99
CFG.DATASET.BIASED_MNIST.ROOT = "./data/biased_mnist"
CFG.DATASET.BIASED_MNIST.IMAGE_SIZE = 28

# --- fb_biased_mnist ---
CFG.DATASET.FB_BIASED_MNIST = CN()
CFG.DATASET.FB_BIASED_MNIST.RATIO = 0
CFG.DATASET.FB_BIASED_MNIST.CORR_BG = 0.9
CFG.DATASET.FB_BIASED_MNIST.CORR_FG = 0.9
CFG.DATASET.FB_BIASED_MNIST.ROOT = "./data/fb_biased_mnist"
CFG.DATASET.FB_BIASED_MNIST.IMAGE_SIZE = 28

# --- utkface ---
CFG.DATASET.UTKFACE = CN()
CFG.DATASET.UTKFACE.BIAS = "race"  # or "age"
CFG.DATASET.UTKFACE.ROOT = "./data/utkface"
CFG.DATASET.UTKFACE.RATIO = 0
CFG.DATASET.UTKFACE.IMAGE_SIZE = 64
CFG.DATASET.UTKFACE.BIAS_ALIGNED = [(1, 1), (0, 0)]

# --- waterbirds ---
CFG.DATASET.WATERBIRDS = CN()
CFG.DATASET.WATERBIRDS.ROOT = "./data/waterbirds"
CFG.DATASET.WATERBIRDS.IMAGE_SIZE = 224

# --- celeba ---
CFG.DATASET.CELEBA = CN()
CFG.DATASET.CELEBA.ROOT = "./data/celeba"
CFG.DATASET.CELEBA.BIAS = "gender"
CFG.DATASET.CELEBA.TARGET = "blonde"  # or "makeup"
CFG.DATASET.CELEBA.RATIO = 0
CFG.DATASET.CELEBA.IMAGE_SIZE = 224
CFG.DATASET.CELEBA.BIAS_ALIGNED = [(0, 0), (1, 1)]

# --- imagenet9 (Background Challenge) ---
CFG.DATASET.IMAGENET9 = CN()
CFG.DATASET.IMAGENET9.ROOT_IMAGENET = "/path/to/imagenet/"  # download ImageNet and set this
CFG.DATASET.IMAGENET9.ROOT_IMAGENET_BG = "./data/imagenet9"
CFG.DATASET.IMAGENET9.IMAGE_SIZE = 224
CFG.DATASET.IMAGENET9.BIAS = "unknown"
CFG.DATASET.IMAGENET9.BENCHMARK_VAL = "mixed_rand"
CFG.DATASET.IMAGENET9.BENCHMARK_TEST = "original"

# --- imagenet9m (synthetic controllable bias benchmark) ---
CFG.DATASET.IMAGENET9M = CN()
CFG.DATASET.IMAGENET9M.ROOT_IMAGENET = "/path/to/imagenet/"  # expects a train/ subdir
CFG.DATASET.IMAGENET9M.MANIFEST_DIR = "./data/imagenet9m"  # reproducibility manifests cache
CFG.DATASET.IMAGENET9M.IMAGE_SIZE = 224
CFG.DATASET.IMAGENET9M.SPLIT_RATIOS = [0.7, 0.1, 0.2]  # train / val / test
CFG.DATASET.IMAGENET9M.SCENARIO = "single"  # "single" or "multi"
CFG.DATASET.IMAGENET9M.CLASSES = [0, 1, 2, 3]  # superclass ids (0..8)
CFG.DATASET.IMAGENET9M.BIAS_TYPE = "jpeg"  # single scenario: "jpeg" or "resize"
CFG.DATASET.IMAGENET9M.CORRELATION = 0.9  # single scenario train bias-aligned fraction
CFG.DATASET.IMAGENET9M.CORRELATION_JPEG = 0.95  # multi scenario
CFG.DATASET.IMAGENET9M.CORRELATION_RESIZE = 0.95  # multi scenario
CFG.DATASET.IMAGENET9M.JPEG_CLASSES = [
    [95, "4:4:4"],
    [75, "4:2:0"],
    [95, "4:2:0"],
    [75, "4:4:4"],
]  # (quality, chroma-subsampling)
CFG.DATASET.IMAGENET9M.RESIZE_CLASSES = [2.0, 0.5, 1.4142136, 0.7071068]  # scale factors

# --- cifar10 ---
CFG.DATASET.CIFAR10 = CN()
CFG.DATASET.CIFAR10.ROOT = "./data/cifar10"
CFG.DATASET.CIFAR10.IMAGE_SIZE = 32
CFG.DATASET.CIFAR10.BIAS = "unknown"

# --- cifar100 ---
CFG.DATASET.CIFAR100 = CN()
CFG.DATASET.CIFAR100.ROOT = "./data/cifar100"
CFG.DATASET.CIFAR100.IMAGE_SIZE = 32
CFG.DATASET.CIFAR100.BIAS = "unknown"

# --- stanford_dogs ---
CFG.DATASET.STANFORD_DOGS = CN()
CFG.DATASET.STANFORD_DOGS.ROOT = "./data/stanford-dogs-dataset"
CFG.DATASET.STANFORD_DOGS.IMAGE_SIZE = 224
CFG.DATASET.STANFORD_DOGS.BIAS = "unknown"

# --- urbancars ---
CFG.DATASET.URBANCARS = CN()
CFG.DATASET.URBANCARS.ROOT = "./data/urbancars"
CFG.DATASET.URBANCARS.IMAGE_SIZE = 224
CFG.DATASET.URBANCARS.BIAS = "bg_cooc_obj"

# ---------------------------------------------------------------------------
# Mitigators
# ---------------------------------------------------------------------------
CFG.MITIGATOR = CN()
CFG.MITIGATOR.TYPE = "erm"  # baseline (ERM / vanilla)

# --- FLAC ---
CFG.MITIGATOR.FLAC = CN()
CFG.MITIGATOR.FLAC.LOSS = CN()
CFG.MITIGATOR.FLAC.LOSS.ALPHA = 110.0
CFG.MITIGATOR.FLAC.LOSS.DELTA = 1.0
CFG.MITIGATOR.FLAC.LOSS.CE_WEIGHT = 1.0
CFG.MITIGATOR.FLAC.BCC_PATH = ""  # single bias-capturing classifier checkpoint
CFG.MITIGATOR.FLAC.BCC_PATHS = []  # or a list of BCC checkpoints (one per bias)

# --- FLAC-B ---
CFG.MITIGATOR.FLACB = CN()
CFG.MITIGATOR.FLACB.BCC_PATH = ""
CFG.MITIGATOR.FLACB.LOSS = CN()
CFG.MITIGATOR.FLACB.LOSS.ALPHA = 110.0
CFG.MITIGATOR.FLACB.LOSS.DELTA = 1.0
CFG.MITIGATOR.FLACB.LOSS.CE_WEIGHT = 1.0

# --- SoftCon ---
CFG.MITIGATOR.SOFTCON = CN()
CFG.MITIGATOR.SOFTCON.BCC_PATH = ""
CFG.MITIGATOR.SOFTCON.WEIGHT = 1000

# --- BAdd ---
CFG.MITIGATOR.BADD = CN()
CFG.MITIGATOR.BADD.M = 1.0
CFG.MITIGATOR.BADD.BCC_PATH = ""
CFG.MITIGATOR.BADD.BCC_PATHS = []

# --- GroupDRO ---
CFG.MITIGATOR.GROUPDRO = CN()
CFG.MITIGATOR.GROUPDRO.ROBUST_STEP_SIZE = 0.01

# --- Spectral Decoupling ---
CFG.MITIGATOR.SD = CN()
CFG.MITIGATOR.SD.COEF = 0.1

# --- EnD ---
CFG.MITIGATOR.END = CN()
CFG.MITIGATOR.END.ALPHA = 1
CFG.MITIGATOR.END.BETA = 1
CFG.MITIGATOR.END.WEIGHT = 1

# --- JTT ---
CFG.MITIGATOR.JTT = CN()
CFG.MITIGATOR.JTT.BIAS_DISCOVERY_EPOCHS = 50
CFG.MITIGATOR.JTT.UPWEIGHT = 100
CFG.MITIGATOR.JTT.BCC_PATH = ""

# --- MAVias ---
CFG.MITIGATOR.MAVIAS = CN()
CFG.MITIGATOR.MAVIAS.TAGGING_MODEL = CN()
CFG.MITIGATOR.MAVIAS.TAGGING_MODEL.TYPE = "ram"
CFG.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE = 384
CFG.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE = 16
CFG.MITIGATOR.MAVIAS.ENCODER = CN()
CFG.MITIGATOR.MAVIAS.ENCODER.TYPE = "clip"
CFG.MITIGATOR.MAVIAS.ENCODER.SIZE = 768
CFG.MITIGATOR.MAVIAS.LLM = CN()
CFG.MITIGATOR.MAVIAS.LLM.TYPE = "llama3"
CFG.MITIGATOR.MAVIAS.LLM.BATCH_SIZE = 100
CFG.MITIGATOR.MAVIAS.LOSS = CN()
CFG.MITIGATOR.MAVIAS.LOSS.ALPHA = 0.1
CFG.MITIGATOR.MAVIAS.LOSS.LAMBDA = 0.8
CFG.MITIGATOR.MAVIAS.PROJNET = CN()
CFG.MITIGATOR.MAVIAS.PROJNET.OPTIM = CN()
CFG.MITIGATOR.MAVIAS.PROJNET.OPTIM.LR = 0.001
CFG.MITIGATOR.MAVIAS.PROJNET.OPTIM.WEIGHT_DECAY = 5e-4
CFG.MITIGATOR.MAVIAS.PROJNET.OPTIM.MOMENTUM = 0.9
CFG.MITIGATOR.MAVIAS.PROJNET.OPTIM.TYPE = "SGD"

# --- MAVias-B (bias-capturing-classifier variant) ---
CFG.MITIGATOR.MAVIASB = CN()
CFG.MITIGATOR.MAVIASB.LOSS = CN()
CFG.MITIGATOR.MAVIASB.LOSS.ALPHA = 0.1
CFG.MITIGATOR.MAVIASB.LOSS.LAMBDA = 0.8
CFG.MITIGATOR.MAVIASB.PROJNET = CN()
CFG.MITIGATOR.MAVIASB.PROJNET.OPTIM = CN()
CFG.MITIGATOR.MAVIASB.PROJNET.OPTIM.LR = 0.001
CFG.MITIGATOR.MAVIASB.PROJNET.OPTIM.WEIGHT_DECAY = 5e-4
CFG.MITIGATOR.MAVIASB.PROJNET.OPTIM.MOMENTUM = 0.9
CFG.MITIGATOR.MAVIASB.PROJNET.OPTIM.TYPE = "SGD"
CFG.MITIGATOR.MAVIASB.BCC_PATH = ""
CFG.MITIGATOR.MAVIASB.BCC_PATHS = []
