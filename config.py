from pathlib import Path
import typing

def get_config() -> typing.Dict:
    """
    Get the configuration for the training process
    Returns:
    -- dict: The configuration for the training process
    Parameters:
    -- batch_size: int: The batch size for the training process
    -- num_epochs: int: The number of epochs for the training process
    -- lr: float: The learning rate for the training process
    -- sequence_length: int: The sequence length for the training process
    -- dimension: int: The text embedding dimension (512 by default)
    -- source_language: str: The source language for the translation
    -- target_language: str: The target language for the translation
    -- model_folder: str: The folder where the model weights are stored
    -- model_basename: str: The base name for the model weights
    -- preload: str: The file path for the weights file to preload
    -- tokenizer_file: str: The file path for the tokenizer file
    -- experiment_name: str: The name of the experiment
    """
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 0.0001,
        "sequence_length": 350,
        "dimensions": 512,
        "datasource": 'opus_books',
        "source_language": "en",
        "target_language": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
    }


def get_weights_file_path(config, epoch: str) -> str:
    """
    Get the file path for the weights file for the given epoch and configuration.
    Returns:
    -- str: The file path for the weights file for the given epoch and configuration.
    """
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(
        Path('.') / model_folder / model_filename
    )


def latest_weights_file_path(config: typing.Dict) -> str:
    """
    Get the file path for the latest weights file for the given configuration.
    Returns:
    -- str: The file path for the latest weights file for the given configuration.
    -- None: If no weights file is found.
    """
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])