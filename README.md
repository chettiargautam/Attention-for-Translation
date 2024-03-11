## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
    - [PyTorch with CUDA](#pytorch-with-cuda)
    - [Setting Up Virtual Environments](#setting-up-virtual-environments)
    - [Install Dependencies](#install-dependencies)
3. [Model Architecture](#model-architecture)
    - [InputEmbedder](#inputembedder)
    - [PositionalEncoder](#positionalencoder)
    - [LayerNormalization](#layernormalization)
    - [FeedForwardLayer](#feedforwardlayer)
    - [MultiHeadAttentionLayer](#multiheadattentionlayer)
    - [ResidualConnection](#residualconnection)
    - [EncoderBlock](#encoderblock)
    - [Encoder](#encoder)
    - [DecoderBlock](#decoderblock)
    - [Decoder](#decoder)
    - [ProjectionLayer](#projectionlayer)
    - [Transformer](#transformer)
    - [build_transformer](#build_transformer)
4. [Dataset Generator](#dataset-generator)
    - [Constructor](#constructor)
    - [Methods](#methods)
        - [__len__](#__len__)
        - [__getitem__](#__getitem__)
        - [causal_mask Function](#causal_mask-function)
5. [Training and Inference](#training-and-inference)
    - [Basic Training](#basic-training)
    - [Customizing Training Parameters](#customizing-training-parameters)
    - [Available Training Arguments](#available-training-arguments)
    - [Validation](#validation)
6. [Acknowledgments](#acknowledgments)


## Introduction

This repo is an attempt to train an Attention model for a Text Translation Task. Most of the code, inspiration and results can be derived from the original source by going to [Coding a Transformer from scratch on PyTorch](https://github.com/hkproj/pytorch-transformer) by Umar Jamil.


## Getting Started

### PyTorch with CUDA (you may need a different GPU support)

You should be able to install PyTorch with CUDA enabled using the official [start locally](https://pytorch.org/get-started/locally/) webpage (it configures the installation line according to your system).  

The goal is to observe the following:  
```shell
python
>>> import torch
>>> torch.cuda.is_available()
True
```

Clone the Repository:  

```shell
git clone https://github.com/chettiargautam/Attention-for-Translation.git
```

Navigate to the project directory:   
```shell
cd attention-for-translation
```

### Setting Up Virtual Environments

- Windows:   
```shell
python -m venv venv
.\venv\Scripts\activate
```

- macOS/Linux:
```shell
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```shell
pip install -r requirements.txt
python setup.py build develop --user
```


## Model Architecture

The `model.py` file defines a Transformer architecture for sequence-to-sequence tasks. Below is an overview of the model's components:

### InputEmbedder

This module is responsible for embedding input tokens into a vector space.

- **Constructor:**
  - `dimensions`: The number of dimensions to embed the input tokens into.
  - `vocabulary_size`: The number of unique tokens in the input.

- **Forward:**
  - Embeds the input tensor into a vector space.
  - Scales output by sqrt(dimensions) as per Vaswani et al. (2017).

### PositionalEncoder

This module encodes the position of the input tokens into the input embeddings.

- **Constructor:**
  - `dimensions`: The number of dimensions to embed the input tokens into.
  - `max_sequence_length`: The maximum length of the input sequence.
  - `dropout`: The dropout rate to apply to the positional encodings.

- **Forward:**
  - Adds the positional encodings to the input tensor.
  - Applies dropout to the output tensor.

### LayerNormalization

This module normalizes the input tensor.

- **Constructor:**
  - `features`: The number of features (dimensions) in the input tensor.
  - `epsilon`: The epsilon value to add to the variance to prevent division by zero.

- **Forward:**
  - Normalizes the input tensor using the gamma and beta parameters.

### FeedForwardLayer

This module projects the input tensor into a higher dimension and then back into the original dimension.

- **Constructor:**
  - `dimension`: The number of dimensions of the input tensor.
  - `projecting_dimension`: The number of dimensions to project the input tensor into.
  - `dropout`: The dropout rate to apply to the output tensor.

- **Forward:**
  - Projects the input tensor into a higher dimension using a linear layer and a ReLU activation function.
  - Projects the input tensor back into the original dimension using a linear layer.
  - Applies dropout to the output tensor.

### MultiHeadAttentionLayer

This module calculates the attention scores for the input tensor.

- **Constructor:**
  - `dimensions`: The number of dimensions of the input tensor.
  - `num_heads`: The number of heads to split the input tensor into.

- **Forward:**
  - Projects the input tensor into the query, key, and value tensors.
  - Splits the query, key, and value tensors into multiple heads.
  - Calculates the attention scores for the input tensor.
  - Projects the attention scores back into the original dimension.

### ResidualConnection

This module adds the input tensor to the output tensor.

- **Constructor:**
  - `features`: The number of features (dimensions) in the input tensor.
  - `dropout`: The dropout rate to apply to the output tensor.

- **Forward:**
  - Adds the input tensor to the output tensor.
  - Applies dropout to the output tensor.

### EncoderBlock

This module is used to encode the input tensor.

- **Constructor:**
  - `features`: The number of features (dimensions) in the input tensor.
  - `self_attention_layer`: The self-attention layer to apply to the input tensor.
  - `feed_forward_layer`: The feed-forward layer to apply to the input tensor.
  - `dropout`: The dropout rate to apply to the output tensor.

- **Forward:**
  - Applies the self-attention layer to the input tensor.
  - Adds the input tensor to the output tensor.
  - Applies the feed-forward layer to the output tensor.
  - Adds the input tensor to the output tensor.

### Encoder

This module is used to encode the input tensor.

- **Constructor:**
  - `features`: The number of features (dimensions) in the input tensor.
  - `layers`: The layers to apply to the input tensor.

- **Forward:**
  - Applies the layers to the input tensor.
  - Applies the layer normalization to the output tensor.

### DecoderBlock

This module is used to decode the input tensor.

- **Constructor:**
  - `features`: The number of features (dimensions) in the input tensor.
  - `self_attention_layer`: The self-attention layer to apply to the input tensor.
  - `cross_attention_layer`: The source attention layer to apply to the input tensor.
  - `feed_forward_layer`: The feed-forward layer to apply to the input tensor.
  - `dropout`: The dropout rate to apply to the output tensor.

- **Forward:**
  - Applies the self-attention layer to the input tensor.
  - Adds the input tensor to the output tensor.
  - Applies the source attention layer to the output tensor.
  - Adds the input tensor to the output tensor.
  - Applies the feed-forward layer to the output tensor.
  - Adds the input tensor to the output tensor.

### Decoder

This module is used to decode the input tensor.

- **Constructor:**
  - `features`: The number of features (dimensions) in the input tensor.
  - `layers`: The layers to apply to the input tensor.

- **Forward:**
  - Applies the layers to the input tensor.
  - Applies the layer normalization to the output tensor.

### ProjectionLayer

This module is used to project the input tensor into a higher dimension.

- **Constructor:**
  - `dimensions`: The number of dimensions of the input tensor.
  - `vocabulary_size`: The number of unique tokens in the input.

- **Forward:**
  - Projects the input tensor into a higher dimension.

### Transformer

This module is used to encode and decode the input tensor.

- **Constructor:**
  - `input_embedder`: The input embedder to embed the input tensor.
  - `target_embedder`: The target embedder to embed the input tensor.
  - `positional_encoder`: The positional encoder to encode the input tensor.
  - `encoder`: The encoder to encode the input tensor.
  - `decoder`: The decoder to decode the input tensor.
  - `projection`: The projection layer to apply to the output tensor.

- **Encode:**
  - Embeds and encodes the input tensor.

- **Decode:**
  - Decodes the input tensor.

- **Project:**
  - Projects the input tensor into a higher dimension.

### build_transformer

This function builds the transformer model with specified parameters.

- **Arguments:**
  - `input_vocabulary_size`: The number of unique tokens in the input.
  - `target_vocabulary_size`: The number of unique tokens in the target.
  - `input_sequence_length`: The maximum length of the input sequence.
  - `target_sequence_length`: The maximum length of the target sequence.
  - `dimensions`: The number of dimensions to embed the input tokens into.
  - `num_layers`: The number of layers to apply to the input tensor.
  - `num_heads`: The number of heads to split the input tensor into.
  - `dropout`: The dropout rate to apply to the output tensor.
  - `linear_projection_dimensions`: The number of dimensions to project the input tensor into.

- **Returns:**
  - A transformer model.


## Dataset Generator
Goes over the functioning of the `dataset.py` script in terms of manipulation of the dataset in order to model it for the task at hand.

### Constructor

- **Parameters:**
  - `dataset`: The dataset to use.
  - `tokenizer_source`: The tokenizer for the source language.
  - `tokenizer_target`: The tokenizer for the target language.
  - `source_language`: The source language.
  - `target_language`: The target language.
  - `sequence_length`: The sequence length to use.

### Methods

#### `__len__`

- **Description:**
  - Get the length of the dataset.

#### `__getitem__`

- **Description:**
  - Get an item from the dataset.
  - Get the source and target texts.
  - Encode the source text.
  - Decode the target text.
  - Pad the encoded and decoded texts.
  - Return the encoded and decoded texts.

- **Parameters:**
  - `index`: The index of the item to retrieve.

- **Returns:**
  - A dictionary containing:
    - `"encoder_input"`: Encoded and padded source text.
    - `"decoder_input"`: Encoded and padded target text.
    - `"encoder_attention_mask"`: Attention mask for the encoder.
    - `"decoder_attention_mask"`: Attention mask for the decoder.
    - `"labels"`: Padded target text with EOS token.
    - `"source_text"`: Original source text.
    - `"target_text"`: Original target text.

### `causal_mask` Function

#### Parameters

- **`length`** (`int`): The length of the causal mask.

#### Returns

- A causal mask for the decoder attention.


## Training and Inference  

To train the model, you can use the provided training script. The hyperparameters can be adjust either in the constants provided in the respective python files, or in the `config.py` file directly.

### Basic Training

```shell
python train.py
```

### Customizing Training Parameters

```shell
python train.py --batch_size 32 --num_epochs 10 --lr 0.001 --sequence_length 50 --dimensions 512 --source_language en --target_language it
```

### Available Training Arguments

- -batch_size: The batch size for training (default: 16).
- -num_epochs: Number of training epochs (default: 5).
- -lr: Learning rate for optimization (default: 0.0001).
- -sequence_length: The sequence length for training (default: 30).
- -dimensions: The text embedding dimension (default: 512).
- -source_language: The source language for translation (default: en).
- -target_language: The target language for translation (default: fr).
- -run_validation: Run validation post-training (default: False).

Adjust these arguments based on your dataset and training requirements in the `config.py` file.  

Note: Also consider altering the number of layers `num_layers` in the Encoder and Decoder to simplify the model, which should make the model less complex and be able to train easier on smaller datasets in lesser time. Also do try to change the `num_examples` parameter if you manage to sufficiently train the model to give out non-gibberish results.

### Validation

To run validation after training, add the --run_validation flag:

```shell
python train.py --run_validation True
```

If that breaks for some reason, then try:

```shell
python test.py
```


## Acknowledgments

Special thanks to Umar Jamil for the foundational work on the transformer model, which served as inspiration for this project. The original source can be found [here](https://github.com/hkproj/pytorch-transformer). He also has put a great video on YouTube explaining the entire working of the Transformer model, Attention mechanism, and another video coding the entire thing from scratch.