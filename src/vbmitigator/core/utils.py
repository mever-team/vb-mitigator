"""General training utilities: seeding, logging, checkpoint I/O."""

import logging
import os
import random
import subprocess
import sys

import numpy as np
import torch


def seed_everything(seed):
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_file, name=None):
    """Return an isolated logger writing INFO records to ``log_file``.

    A dedicated (non-root) logger keyed by ``name`` is used, with propagation
    disabled and any pre-existing handlers cleared. This prevents handler
    accumulation and duplicated log lines when several trainers are created in
    the same process (e.g. the UI launching runs, or the test suite).
    """
    logger = logging.getLogger(name or log_file)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger


def log_msg(msg, mode="INFO", logger=None):
    """Log ``msg`` at level ``mode`` (INFO | TRAIN | EVAL), colorized when printed."""
    color_map = {"INFO": 36, "TRAIN": 32, "EVAL": 31}
    if logger:
        if mode == "INFO":
            logger.info(msg)
        else:
            logger.info("\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg))
    else:
        print("\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg))
    return msg


def save_checkpoint(obj, path):
    """Serialize ``obj`` (e.g. a training state dict) to ``path``."""
    with open(path, "wb") as f:
        torch.save(obj, f)


def load_checkpoint(path):
    """Load a checkpoint saved by :func:`save_checkpoint` (onto CPU).

    ``weights_only=False`` is explicit: our checkpoints bundle optimizer and
    scheduler state (not just tensors), and it is future-proof against the
    upcoming change of PyTorch's default. Only load checkpoints you trust.
    """
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu", weights_only=False)


def load_ollama_docker(llm_name):
    """Ensure an ``ollama`` docker container is running and pull ``llm_name``.

    Used only by the MAVias / erm_tags LLM tag-relevance pipeline.
    """
    try:
        running = subprocess.run(
            ["docker", "ps", "-q", "-f", "name=ollama"],
            capture_output=True,
            text=True,
        )
        if running.stdout.strip():
            print("Container 'ollama' is already running.")
        else:
            stopped = subprocess.run(
                ["docker", "ps", "-aq", "-f", "name=ollama"],
                capture_output=True,
                text=True,
            )
            if stopped.stdout.strip():
                print("Container 'ollama' exists but is stopped. Starting it...")
                subprocess.run(["docker", "start", "ollama"])
            else:
                print("Container 'ollama' does not exist. Creating it...")
                subprocess.run(
                    "docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 "
                    "--name ollama ollama/ollama",
                    shell=True,
                )
        print("Executing LLM in the running container...")
        subprocess.run(f"docker exec -it ollama ollama run {llm_name}", shell=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
