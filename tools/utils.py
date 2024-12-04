import torch
import logging


def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def log_msg(msg, mode="INFO", logger=None):
    """
    Logs a message with a specific mode and color.

    Args:
        msg (str): The message to be logged.
        mode (str, optional): The mode of the log message. Defaults to "INFO".
            Available modes are:
            - "INFO": Informational messages (default, color code 36).
            - "TRAIN": Training messages (color code 32).
            - "EVAL": Evaluation messages (color code 31).
        logger (logging.Logger, optional): The logger to use for logging the message. If None, the message will be printed with ANSI color codes.

    Returns:
        None
    """
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "EVAL": 31,
    }
    if logger:
        if mode == "INFO":
            logger.info(msg)
        elif mode == "TRAIN":
            logger.info("\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg))
        elif mode == "EVAL":
            logger.info("\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg))
    else:
        msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
        print(msg)


def save_checkpoint(obj, path):
    """
    Save a checkpoint object to a specified file path.

    Args:
        obj (Any): The object to be saved, typically a model state dictionary or other checkpoint data.
        path (str): The file path where the object will be saved.

    Returns:
        None
    """
    with open(path, "wb") as f:
        torch.save(obj, f)


def load_checkpoint(path):
    """
    Load a checkpoint from a given file path.

    Args:
        path (str): The file path to the checkpoint file.

    Returns:
        dict: The loaded checkpoint data.

    Example:
        checkpoint = load_checkpoint("/path/to/checkpoint.pth")
    """
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu")
