from dataset import *
from config import *
from train import *


"""CONSTANT VALUES"""
DROPOUT_CONSTANT = 0.1
TEXT_EMBEDDING_DIMENSIONS = 512
LINEAR_PROJECTION_DIMENSIONS = 2048
VOCABULARY_SIZE = 1000
NUM_HEADS = 8
INF = 1e8


def main():
    """
    Main function for the validation process:
    - Get the configuration
    - Get the dataset
    - Get the model
    - Run the validation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()
    _, val_dataloader, source_tokenizer, target_tokenizer = get_dataset(config)
    model = get_model(config, source_tokenizer.get_vocab_size(), target_tokenizer.get_vocab_size()).to(device)
    run_validation(
        model = model, 
        validation_ds = val_dataloader, 
        source_tokenizer = source_tokenizer, 
        target_tokenizer = target_tokenizer, 
        max_length = config['sequence_length'], 
        device = device, 
        print_message = print, 
        global_step = 0,
        num_examples = 1
    )


if __name__ == "__main__":
    main()