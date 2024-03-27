from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

import torch, warnings, os, argparse
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


"""CONSTANT VALUES"""
DROPOUT_CONSTANT = 0.1
TEXT_EMBEDDING_DIMENSIONS = 512
LINEAR_PROJECTION_DIMENSIONS = 2048
VOCABULARY_SIZE = 1000
NUM_HEADS = 8
NUM_LAYERS = 2
INF = 1e8


def greedy_decode(model, source, source_mask, source_tokenizer, target_tokenizer, max_length, device):
    """
    Greedy Decode:
    - Finds the most likely next token at each step and appends it to the decoder input.
    - Just picks the token with the max probabilitiesability at each step as the next token.
    - May not give the ideal output, but it's fast and simple. For better results, use beam search.
    Parameters:
    - model: The transformer model
    - source: The input sentence
    - source_mask: The mask for the input sentence
    - source_tokenizer: The tokenizer for the source language
    - target_tokenizer: The tokenizer for the target language
    - max_length: The maximum length of the output sentence
    - device: The device to run the model on
    Shapes:
    - source: (1, sequence_length, dimensions)
    - source_mask: (1, 1, sequence_length, sequence_length)
    Returns: 
    - The output sentence
    """
    start_of_sequence_index = target_tokenizer.token_to_id('[SOS]')
    end_of_sequence_index = target_tokenizer.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    
    decoder_input = torch.empty(1, 1).fill_(start_of_sequence_index).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_length:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        probabilities = model.project(out[:, -1])
        _, next_word = torch.max(probabilities, dim=1)

        decoder_input = torch.cat(
            [
                decoder_input, 
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)
            ], 
            dim=1
        )

        if next_word == end_of_sequence_index:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, source_tokenizer, target_tokenizer, max_length, device, print_message, global_step, num_examples=2):
    """
    Run Validation:
    - Runs the model on the validation dataset and prints the source, target and predicted sentences.
    Parameters:
    - model: The transformer model
    - validation_ds: The validation dataset
    - source_tokenizer: The tokenizer for the source language
    - target_tokenizer: The tokenizer for the target language
    - max_length: The maximum length of the output sentence
    - device: The device to run the model on
    - print_message: A function to print messages
    - global_step: The global step
    - num_examples: The number of examples to print
    Shapes:
    - source: (1, sequence_length, dimensions)
    - source_mask: (1, 1, sequence_length, sequence_length)
    Returns:
    - Inference
    """
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_attention_mask"].to(device)

            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, source_tokenizer, target_tokenizer, max_length, device)

            source_text = batch["source_text"][0]
            target_text = batch["target_text"][0]
            model_out_text = target_tokenizer.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            print_message('-'*console_width)
            print_message(f"{f'SOURCE: ':>12}{source_text}")
            print_message(f"{f'TARGET: ':>12}{target_text}")
            print_message(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_message('-'*console_width)
                break


def get_all_sentences(ds, language):
    """
    Get All Sentences:
    - A generator to get all sentences in a dataset
    Parameters:
    - ds: The dataset
    - language: The language
    Returns:
    - A generator
    """
    for item in ds:
        yield item['translation'][language]


def get_or_build_tokenizer(config, ds, language):
    """
    Get or Build Tokenizer:
    - Gets the tokenizer if it exists, otherwise builds it and saves it.
    Parameters:
    - config: The configuration
    - ds: The dataset
    - language: The language
    Returns:
    - The tokenizer
    Refs:
    - https://huggingface.co/docs/tokenizers/quicktour
    """
    tokenizer_path = Path(config['tokenizer_file'].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_dataset(config):
    """
    Get Dataset:
    - Loads the dataset, builds the tokenizers and returns the dataloaders.
    Parameters:
    - config: The configuration
    Returns:
    - The training and validation dataloaders
    """
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['source_language']}-{config['target_language']}", split='train')

    source_tokenizer = get_or_build_tokenizer(config, ds_raw, config['source_language'])
    target_tokenizer = get_or_build_tokenizer(config, ds_raw, config['target_language'])

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw, 
        source_tokenizer, 
        target_tokenizer, 
        config['source_language'], 
        config['target_language'], 
        config['sequence_length'],
    )

    val_ds = BilingualDataset(
        val_ds_raw, 
        source_tokenizer, 
        target_tokenizer, 
        config['source_language'], 
        config['target_language'], 
        config['sequence_length'],
    )

    max_length_src = 0
    max_length_tgt = 0

    for item in ds_raw:
        src_ids = source_tokenizer.encode(item['translation'][config['source_language']]).ids
        tgt_ids = target_tokenizer.encode(item['translation'][config['target_language']]).ids
        max_length_src = max(max_length_src, len(src_ids))
        max_length_tgt = max(max_length_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_length_src}')
    print(f'Max length of target sentence: {max_length_tgt}')
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, source_tokenizer, target_tokenizer


def get_model(config, vocab_src_len, vocab_tgt_len):
    """
    Get Model:
    - Builds the transformer model
    Parameters:
    - config: The configuration
    - vocab_src_len: The length of the source vocabulary
    - vocab_tgt_len: The length of the target vocabulary
    Returns:
    - The transformer model
    """
    return build_transformer(
        input_vocabulary_size = vocab_src_len, 
        target_vocabulary_size = vocab_tgt_len, 
        input_sequence_length = config["sequence_length"], 
        target_sequence_length = config['sequence_length'], 
        dimensions = config['dimensions'],
        num_layers = NUM_LAYERS,
        num_heads = NUM_HEADS,
        dropout = DROPOUT_CONSTANT,
        linear_projection_dimensions = LINEAR_PROJECTION_DIMENSIONS,
    )


def train_model(config):
    """
    Train Model:
    - Trains the transformer model
    Parameters:
    - config: The configuration
    Steps:
    - Get the dataset
    - Get the model
    - Define the optimizer
    - Define the loss function
    - For each epoch:
        - For each batch:
            - Run the tensors through the encoder, decoder and the projection layer
            - Compare the output with the label
            - Compute the loss using a simple cross entropy
            - Backpropagate the loss
            - Update the weights
        - Run validation at the end of every epoch
        - Save the model at the end of every epoch
    """
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    device = torch.device(device)

    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, source_tokenizer, target_tokenizer = get_dataset(config)
    model = get_model(config, source_tokenizer.get_vocab_size(), target_tokenizer.get_vocab_size()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=source_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_attention_mask'].to(device)
            decoder_mask = batch['decoder_attention_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, sequence_length, dimensions)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch_size, sequence_length, dimensions)
            proj_output = model.project(decoder_output) # (batch_size, sequence_length, vocab_size)

            label = batch['labels'].to(device) # (batch_size, sequence_length)

            loss = loss_fn(proj_output.view(-1, target_tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            torch.cuda.empty_cache()

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, 
        model_filename)


def main():
    """
    Main:
    - The main function
    Flow:
    - Check if the user is running the file as it is.
    - If it is run directly without additional system arguments, take default config values from config.py and train the model.
    - If it is run with additional system arguments, take the config values from the system arguments and train the model, and still keep defaults from the config file for missing arguments.
    """
    config = get_config()

    parser = argparse.ArgumentParser(description='Train a transformer model for translation')
    parser.add_argument(
        '--batch_size', 
        type=int, 
        help='The batch size for the training process',
        default=config['batch_size']
    )
    parser.add_argument(
        '--num_epochs', 
        type=int, 
        help='The number of epochs for the training process',
        default=config['num_epochs']
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        help='The learning rate for the training process',
        default=config['lr']
    )
    parser.add_argument(
        '--sequence_length', 
        type=int, 
        help='The sequence length for the training process',
        default=config['sequence_length']
    )
    parser.add_argument(
        '--dimensions', 
        type=int, 
        help='The text embedding dimension (512 by default)',
        default=config['dimensions']
    )
    parser.add_argument(
        '--source_language', 
        type=str, 
        help='The source language for the translation',
        default=config['source_language']
    )
    parser.add_argument(
        '--target_language', 
        type=str, 
        help='The target language for the translation',
        default=config['target_language']
    )
    parser.add_argument(
        '--run_validation', 
        type=bool, 
        help='Run validation post training',
        default=False
    )
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    """
    Updating the config if the system arguments are provided
    """
    if len(vars(args)) > 0:
        config.update(vars(args))
        
    """
    The model is trained with the /updated config
    """
    train_model(config)

    """
    If the system argument run_validation is True, the model is validated
    """
    if args.run_validation:
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


if __name__ == '__main__':
    """
    Change the directory to the current file's directory to install files within the repository no matter where the file is called from
    """
    if os.getcwd().split('/')[-1] != 'Attention for Translation':
        current_file_path = os.path.abspath(__file__)
        current_dir_path = os.path.dirname(current_file_path)
        os.chdir(current_dir_path)
    """
    Most warnings are ignored to keep the output clean
    """
    warnings.filterwarnings("ignore")
    
    main()