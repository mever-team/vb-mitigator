import os
import sys
import subprocess
import torch
import logging
import numpy as np
import random

# this function guarantees reproductivity
# other packages also support seed options, you can add to this function
def seed_everything(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

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


def load_ollama_docker(llm_name):
    """
    Load and run an Ollama Docker container with the specified LLM.

    Args:
        llm_name (str): The name of the LLM to run inside the Docker container.

    Returns:
        None
    """
    # Command to run the Docker container
    # Check if the container exists
    try:
        # Check if the container exists and is running
        existing_containers = subprocess.run(
            ["docker", "ps", "-q", "-f", "name=ollama"],
            capture_output=True,
            text=True,
        )

        if existing_containers.stdout.strip():  # Container is running
            print("Container 'ollama' is already running.")
        else:
            # Check if the container exists (stopped)
            existing_containers_stopped = subprocess.run(
                ["docker", "ps", "-aq", "-f", "name=ollama"],
                capture_output=True,
                text=True,
            )

            if (
                existing_containers_stopped.stdout.strip()
            ):  # Container exists but is stopped
                print("Container 'ollama' exists but is not running. Starting it...")
                subprocess.run(["docker", "start", "ollama"])
            else:  # Container does not exist, create it
                print("Container 'ollama' does not exist. Creating it...")
                run_docker_command = f"docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama"
                subprocess.run(run_docker_command, shell=True)
        # At this point, the container should be running
        print("Executing LLM in the running container...")
        exec_docker_command = f"docker exec -it ollama ollama run {llm_name}"
        subprocess.run(exec_docker_command, shell=True)

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
    return


def load_ollama(llm_name):
    """
    Ensures that the Ollama tool is installed, starts the Ollama server, and pulls the specified LLM model.

    Parameters:
    llm_name (str): The name of the LLM model to pull.

    This function performs the following steps:
    1. Checks if the Ollama tool is installed. If not, it installs Ollama using a shell script.
    2. Starts the Ollama server.
    3. Pulls the specified LLM model using the Ollama tool.

    If any step fails, an appropriate error message is printed and the function exits.

    Returns:
    None
    """
    if not os.path.exists("/usr/local/bin/ollama"):
        print("Ollama not found, installing...")
        try:
            # Use subprocess.run instead of os.system for better error handling
            subprocess.run(
                "curl -fsSL https://ollama.com/install.sh | sh",
                shell=True,
                check=True,
            )
            print("Ollama installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error during Ollama installation: {e}")
            return  # Exit if installation fails

    # Serve the Ollama model
    print("Starting Ollama server...")
    try:
        subprocess.Popen("ollama serve", shell=True)
        print("Ollama server started.")
    except Exception as e:
        print(f"Error starting Ollama server: {e}")
        return  # Exit if server fails to start

    # Pull the specified LLM model
    print(f"Pulling model: {llm_name}...")
    try:
        subprocess.run(f"ollama pull {llm_name}", shell=True, check=True)
        print(f"Model '{llm_name}' pulled successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error pulling model '{llm_name}': {e}")
    return
