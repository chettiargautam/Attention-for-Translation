import torch, numpy, typing

"""CONSTANT VALUES"""
DROPOUT_CONSTANT = 0.1
TEXT_EMBEDDING_DIMENSIONS = 512
LINEAR_PROJECTION_DIMENSIONS = 2048
VOCABULARY_SIZE = 1000
NUM_HEADS = 8
INF = 1e8

class InputEmbedder(torch.nn.Module):
    def __init__(self, dimensions: int, vocabulary_size: int) -> None:
        """
        Constructor:
            Initializes the Embedder module. This module is used to embed the input tokens into a vector space.
        Arguments:
            dimensions: int - The number of dimensions to embed the input tokens into.
            vocabulary_size: int - The number of unique tokens in the input.
        """
        super(InputEmbedder, self).__init__()
        self.dimensions = dimensions
        self.vocabulary_size = vocabulary_size
        self.embedding_function = torch.nn.Embedding(vocabulary_size, dimensions)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward:
            - Embeds the input tensor into a vector space. This is done by looking up the embedding for each token in the input tensor.
            - Scales output by sqrt(dimensions) as per Vaswani et al. (2017).
            -- Takes in an input tensor in R and returns an output tensor in R^d.
            -- Takes in an input tensor of shape (batch_size, sequence_length) and returns an output tensor of shape (batch_size, sequence_length, dimensions).
        Arguments:
            input_tensor: torch.Tensor - The input tensor to embed.
        Shape Transformation:
            - (batch_size, sequence_length) -> (batch_size, sequence_length, dimensions)
        """
        return self.embedding_function(input_tensor) * numpy.sqrt(self.dimensions)


class PositionalEncoder(torch.nn.Module):
    def __init__(self, dimensions: int, max_sequence_length: int, dropout: int = DROPOUT_CONSTANT) -> None:
        """
        Constructor:
            Initializes the PositionalEncoder module. This module is used to encode the position of the input tokens into the input embeddings.
        Arguments:
            dimensions: int - The number of dimensions to embed the input tokens into.
            max_sequence_length: int - The maximum length of the input sequence. The PE's will be computed for this length.
            dropout: int - The dropout rate to apply to the PEs.
        """
        super(PositionalEncoder, self).__init__()
        self.dimensions = dimensions
        self.max_sequence_length = max_sequence_length
        self.dropout = torch.nn.Dropout(dropout)

        """
        - Create a matrix of shape (max_sequence_length, dimensions) to store the positional encodings.
        - For each position in the sequence, compute the positional encoding for each dimension.
        - Store the positional encodings in the matrix.
        - Implement the positional encodings as per Vaswani et al. (2017).
        """
        positional_encodings = numpy.zeros((max_sequence_length, dimensions))
        pos = torch.arange(0, max_sequence_length).unsqueeze(1)
        denominator = numpy.power(10000, 2 * numpy.arange(0, dimensions, 2) / dimensions)
        positional_encodings[:, 0::2] = numpy.sin(pos / denominator)
        positional_encodings[:, 1::2] = numpy.cos(pos / denominator)
        positional_encodings = torch.tensor(positional_encodings).float()
        positional_encodings = positional_encodings.unsqueeze(0)
        self.register_buffer('positional_encodings', positional_encodings)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward:
            - Adds the positional encodings to the input tensor.
            - Applies dropout to the output tensor.
            -- Takes in an input tensor in R^d and returns an output tensor in R^d.
            -- Takes in an input tensor of shape (batch_size, sequence_length, dimensions) and returns an output tensor of shape (batch_size, sequence_length, dimensions).
        Arguments:
            input_tensor: torch.Tensor - The input tensor to add positional encodings to.
        Shape Transformation:
            - (batch_size, sequence_length, dimensions) -> (batch_size, sequence_length, dimensions)
        """
        input_tensor += self.positional_encodings[:, :input_tensor.size(1), :].requires_grad_(False)
        return self.dropout(input_tensor)


class LayerNormalization(torch.nn.Module):
    def __init__(self, features: int, epsilon: float = 1e-7) -> None:
        """
        Constructor:
            Initializes the LayerNormalization module. This module is used to normalize the input tensor.
        Arguments:
            epsilon: float - The epsilon value to add to the variance to prevent division by zero.
        """
        super(LayerNormalization, self).__init__()
        self.epsilon = epsilon
        self.gamma = torch.nn.Parameter(torch.ones(features))
        self.beta = torch.nn.Parameter(torch.zeros(features))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward:
            - Normalizes the input tensor using the gamma and beta parameters.
            -- Takes in an input tensor in R^d and returns an output tensor in R^d.
            -- Takes in an input tensor of shape (batch_size, sequence_length, dimensions) and returns an output tensor of shape (batch_size, sequence_length, dimensions).
            -- Assumes input has batch, the mean and standard deviation is calculated over the last dimension leaving the batch dimension.
        Arguments:
            input_tensor: torch.Tensor - The input tensor to normalize.
        Shape Transformation:
            - (batch_size, sequence_length, dimensions) -> (batch_size, sequence_length, dimensions)
        """
        self.mean = input_tensor.mean(-1, keepdim=True)
        self.standard_deviation = input_tensor.std(-1, keepdim=True)
        return self.gamma * (input_tensor - self.mean) / (self.standard_deviation + self.epsilon) + self.beta


class FeedForwardLayer(torch.nn.Module):
    def __init__(self, dimension: int, projecting_dimension: int, dropout: int) -> None:
        """
        Constructor:
            Initializes the FeedForwardLayer module. This module is used to project the input tensor into a higher dimension and then back into the original dimension.
        Arguments:
            dimension: int - The number of dimensions of the input tensor.
            projecting_dimension: int - The number of dimensions to project the input tensor into.
            dropout: int - The dropout rate to apply to the output tensor.
        """
        super(FeedForwardLayer, self).__init__()
        self.dimension = dimension
        self.projecting_dimension = projecting_dimension
        self.LinearLayer1 = torch.nn.Linear(dimension, projecting_dimension)
        self.LinearLayer2 = torch.nn.Linear(projecting_dimension, dimension)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward:
            - Projects the input tensor into a higher dimension using a linear layer and a ReLU activation function.
            - Projects the input tensor back into the original dimension using a linear layer.
            - Applies dropout to the output tensor.
            -- Takes in an input tensor in R^d and returns an output tensor in R^d.
            -- Takes in an input tensor of shape (batch_size, sequence_length, dimensions) and returns an output tensor of shape (batch_size, sequence_length, dimensions).
        Arguments:
            input_tensor: torch.Tensor - The input tensor to project and then project back.
        Shape Transformation:
            - (batch_size, sequence_length, dimensions) -> (batch_size, sequence_length, projecting_dimension)
            - (batch_size, sequence_length, projecting_dimension) -> (batch_size, sequence_length, projecting_dimension)
            - (batch_size, sequence_length, projecting_dimension) -> (batch_size, sequence_length, dimensions)
        """
        input_tensor = self.LinearLayer1(input_tensor)
        input_tensor = self.relu(input_tensor)
        input_tensor = self.LinearLayer2(input_tensor)
        return self.dropout(input_tensor)


class MultiHeadAttentionLayer(torch.nn.Module):
    def __init__(self, dimensions: int, num_heads: int, dropout: int) -> None:
        """
        Constructor:
            Initializes the MultiHeadAttentionLayer module. This module is used to calculate the attention scores for the input tensor.
        Arguments:
            dimensions: int - The number of dimensions of the input tensor.
            num_heads: int - The number of heads to split the input tensor into.
        """
        super(MultiHeadAttentionLayer, self).__init__()
        self.dimensions = dimensions
        self.num_heads = num_heads
        self.dropout = torch.nn.Dropout(dropout)

        assert dimensions % num_heads == 0, "The number of dimensions must be divisible by the number of heads."
        self.head_dimension = dimensions // num_heads

        """
        Intialize the linear layers for the query, key, and value projections.
        """
        self.query_projection = torch.nn.Linear(dimensions, dimensions)
        self.key_projection = torch.nn.Linear(dimensions, dimensions)
        self.value_projection = torch.nn.Linear(dimensions, dimensions)
        self.output_projection = torch.nn.Linear(dimensions, dimensions)

    @staticmethod
    def Attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None, dropout: int = DROPOUT_CONSTANT) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Attention:
            - Calculates the attention scores for the input tensor.
            - Splits the input tensor into multiple heads.
            - Scales the input tensor by the square root of the number of heads.
            - Calculates the attention scores for the input tensor.
            - Applies dropout to the attention scores.
            - Concatenates the attention scores for the multiple heads.
            - Projects the attention scores back into the original dimension.
            -- Takes in query, key, and value tensors in R^d and returns an output tensor in R^d.
            -- Takes in query, key, and value tensors of shape (batch_size, sequence_length, dimensions) and returns an output tensor of shape (batch_size, sequence_length, dimensions).
        Arguments:
            query: torch.Tensor - The query tensor to calculate the attention scores for.
            key: torch.Tensor - The key tensor to calculate the attention scores for.
            value: torch.Tensor - The value tensor to calculate the attention scores for.
            mask: torch.Tensor - The mask tensor to apply to the attention scores.
            dropout: int - The dropout rate to apply to the attention scores.
        Linear Algebra:
            - input_tensor: (batch_size, sequence_length, dimensions)
            - query, key and value: (batch_size, sequence_length, dimensions)
            - attention_scores: (batch_size, sequence_length, sequence_length)
        """
        dimension = query.size(-1)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / numpy.sqrt(dimension)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -INF)
        
        attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = torch.nn.Dropout(dropout)(attention_scores)
        return torch.matmul(attention_scores, value), attention_scores
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward:
            - Projects the input tensor into the query, key, and value tensors.
            - Splits the query, key, and value tensors into multiple heads.
            - Calculates the attention scores for the input tensor.
            - Concatenates the attention scores for the multiple heads.
            - Projects the attention scores back into the original dimension.
            -- Takes in query, key, and value tensors in R^d and returns an output tensor in R^d.
            -- Takes in query, key, and value tensors of shape (batch_size, sequence_length, dimensions) and returns an output tensor of shape (batch_size, sequence_length, dimensions).
        Arguments:
            query: torch.Tensor - The query tensor to calculate the attention scores for.
            key: torch.Tensor - The key tensor to calculate the attention scores for.
            value: torch.Tensor - The value tensor to calculate the attention scores for.
            mask: torch.Tensor - The mask tensor to apply to the attention scores.
        """
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)

        query = query.view(query.size(0), query.size(1), self.num_heads, self.head_dimension).transpose(1, 2)
        key = key.view(key.size(0), key.size(1), self.num_heads, self.head_dimension).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), self.num_heads, self.head_dimension).transpose(1, 2)

        enriched_values, self.attention_scores = MultiHeadAttentionLayer.Attention(query, key, value, mask)

        enriched_values = enriched_values.transpose(1, 2).contiguous().view(enriched_values.size(0), -1, self.dimensions)

        enriched_values = self.output_projection(enriched_values)

        return enriched_values


class ResidualConnection(torch.nn.Module):
    def __init__(self, features: int, dropout: float = DROPOUT_CONSTANT) -> None:
        """
        Constructor:
            Initializes the ResidualConnection module. This module is used to add the input tensor to the output tensor.
        Arguments:
            dropout: float - The dropout rate to apply to the output tensor.
        """
        super(ResidualConnection, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_normalization = LayerNormalization(features)

    def forward(self, input_tensor: torch.Tensor, networkLayer: torch.nn.Module) -> torch.Tensor:
        """
        Forward:
            - Adds the input tensor to the output tensor.
            - Applies dropout to the output tensor.
            -- Takes in an input tensor in R^d and returns an output tensor in R^d.
            -- Takes in an input tensor of shape (batch_size, sequence_length, dimensions) and returns an output tensor of shape (batch_size, sequence_length, dimensions).
        Arguments:
            input_tensor: torch.Tensor - The input tensor to add to the output tensor.
            networkLayer: torch.nn.Module - The network layer to apply to the input tensor.
        Shape Transformation:
            - (batch_size, sequence_length, dimensions) -> (batch_size, sequence_length, dimensions)
        """
        return input_tensor + self.dropout(self.layer_normalization(networkLayer(input_tensor)))


class EncoderBlock(torch.nn.Module):
    def __init__(self, features: int, self_attention_layer: MultiHeadAttentionLayer, feed_forward_layer: FeedForwardLayer, dropout: float = DROPOUT_CONSTANT) -> None:
        """
        Constructor:
            Initializes the EncoderBlock module. This module is used to encode the input tensor.
        Arguments:
            self_attention_layer: MultiHeadAttentionLayer - The self attention layer to apply to the input tensor.
            feed_forward_layer: FeedForwardLayer - The feed forward layer to apply to the input tensor.
            dropout: float - The dropout rate to apply to the output tensor.
        """
        super(EncoderBlock, self).__init__()
        self.self_attention_layer = self_attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.residual_connections = torch.nn.ModuleList(
            [
                ResidualConnection(features, dropout) for _ in range(2)
            ]
        )
        
    def forward(self, input_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward:
            - Applies the self attention layer to the input tensor.
            - Adds the input tensor to the output tensor.
            - Applies the feed forward layer to the output tensor.
            - Adds the input tensor to the output tensor.
            -- Takes in an input tensor in R^d and returns an output tensor in R^d.
            -- Takes in an input tensor of shape (batch_size, sequence_length, dimensions) and returns an output tensor of shape (batch_size, sequence_length, dimensions).
        Arguments:
            input_tensor: torch.Tensor - The input tensor to encode.
            mask: torch.Tensor - The mask tensor to apply to the attention scores.
        """
        input_tensor = self.residual_connections[0](input_tensor, lambda input_tensor: self.self_attention_layer(input_tensor, input_tensor, input_tensor, mask))
        return self.residual_connections[1](input_tensor, self.feed_forward_layer)
    

class Encoder(torch.nn.Module):
    def __init__(self, features: int, layers: torch.nn.ModuleList) -> None:
        """
        Constructor:
            Initializes the Encoder module. This module is used to encode the input tensor.
        Arguments:
            layers: torch.nn.ModuleList - The layers to apply to the input tensor.
        """
        super(Encoder, self).__init__()
        self.layers = layers
        self.layer_normalization = LayerNormalization(features)

    def forward(self, input_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward:
            - Applies the layers to the input tensor.
            - Applies the layer normalization to the output tensor.
            -- Takes in an input tensor in R^d and returns an output tensor in R^d.
            -- Takes in an input tensor of shape (batch_size, sequence_length, dimensions) and returns an output tensor of shape (batch_size, sequence_length, dimensions).
        Arguments:
            input_tensor: torch.Tensor - The input tensor to encode.
            mask: torch.Tensor - The mask tensor to apply to the attention scores.
        """
        for layer in self.layers:
            input_tensor = layer(input_tensor, mask)
        return self.layer_normalization(input_tensor)


class DecoderBlock(torch.nn.Module):
    def __init__(self, features: int, self_attention_layer: MultiHeadAttentionLayer, cross_attention_layer: MultiHeadAttentionLayer, feed_forward_layer: FeedForwardLayer, dropout: float = DROPOUT_CONSTANT) -> None:
        """
        Constructor:
            Initializes the DecoderBlock module. This module is used to decode the input tensor.
        Arguments:
            self_attention_layer: MultiHeadAttentionLayer - The self attention layer to apply to the input tensor.
            cross_attention_layer: MultiHeadAttentionLayer - The source attention layer to apply to the input tensor.
            feed_forward_layer: FeedForwardLayer - The feed forward layer to apply to the input tensor.
            dropout: float - The dropout rate to apply to the output tensor.
        """
        super(DecoderBlock, self).__init__()
        self.self_attention_layer = self_attention_layer
        self.cross_attention_layer = cross_attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.residual_connections = torch.nn.ModuleList(
            [
                ResidualConnection(features, dropout) for _ in range(3)
            ]
        )
    
    def forward(self, input_tensor: torch.Tensor, encoder_output: torch.Tensor, source_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward:
            - Applies the self attention layer to the input tensor.
            - Adds the input tensor to the output tensor.
            - Applies the source attention layer to the output tensor.
            - Adds the input tensor to the output tensor.
            - Applies the feed forward layer to the output tensor.
            - Adds the input tensor to the output tensor.
            -- Takes in an input tensor in R^d and returns an output tensor in R^d.
            -- Takes in an input tensor of shape (batch_size, sequence_length, dimensions) and returns an output tensor of shape (batch_size, sequence_length, dimensions).
        Arguments:
            input_tensor: torch.Tensor - The input tensor to decode.
            encoder_output: torch.Tensor - The output tensor from the encoder.
            source_mask: torch.Tensor - The mask tensor to apply to the attention scores for the source tensor.
            target_mask: torch.Tensor - The mask tensor to apply to the attention scores for the target tensor.
        """
        input_tensor = self.residual_connections[0](input_tensor, lambda input_tensor: self.self_attention_layer(input_tensor, input_tensor, input_tensor, target_mask))
        input_tensor = self.residual_connections[1](input_tensor, lambda input_tensor: self.cross_attention_layer(input_tensor, encoder_output, encoder_output, source_mask))
        input_tensor = self.residual_connections[2](input_tensor, self.feed_forward_layer)
        return input_tensor
        

class Decoder(torch.nn.Module):
    def __init__(self, features: int, layers: torch.nn.ModuleList) -> None:
        """
        Constructor:
            Initializes the Decoder module. This module is used to decode the input tensor.
        Arguments:
            layers: torch.nn.ModuleList - The layers to apply to the input tensor.
        """
        super(Decoder, self).__init__()
        self.layers = layers
        self.layer_normalization = LayerNormalization(features)

    def forward(self, input_tensor: torch.Tensor, encoder_output: torch.Tensor, source_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward:
            - Applies the layers to the input tensor.
            - Applies the layer normalization to the output tensor.
            -- Takes in an input tensor in R^d and returns an output tensor in R^d.
            -- Takes in an input tensor of shape (batch_size, sequence_length, dimensions) and returns an output tensor of shape (batch_size, sequence_length, dimensions).
        Arguments:
            input_tensor: torch.Tensor - The input tensor to decode.
            encoder_output: torch.Tensor - The output tensor from the encoder.
            source_mask: torch.Tensor - The mask tensor to apply to the attention scores for the source tensor.
            target_mask: torch.Tensor - The mask tensor to apply to the attention scores for the target tensor.
        """
        for layer in self.layers:
            input_tensor = layer(input_tensor, encoder_output, source_mask, target_mask)
        return self.layer_normalization(input_tensor)


class ProjectionLayer(torch.nn.Module):
    def __init__(self, dimensions: int, vocabulary_size: int) -> None:
        """
        Constructor:
            Initializes the ProjectionLayer module. This module is used to project the input tensor into a higher dimension.
        Arguments:
            dimensions: int - The number of dimensions of the input tensor.
            vocabulary_size: int - The number of unique tokens in the input.
        """
        super(ProjectionLayer, self).__init__()
        self.dimensions = dimensions
        self.vocabulary_size = vocabulary_size
        self.linear_layer = torch.nn.Linear(dimensions, vocabulary_size)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward:
            - Projects the input tensor into a higher dimension using a linear layer.
            -- Takes in an input tensor in R^d and returns an output tensor in R^v.
            -- Takes in an input tensor of shape (batch_size, sequence_length, dimensions) and returns an output tensor of shape (batch_size, sequence_length, vocabulary_size).
        Arguments:
            input_tensor: torch.Tensor - The input tensor to project.
        """
        return self.softmax(self.linear_layer(input_tensor))


class Transformer(torch.nn.Module):
    def __init__(self, input_embedder: InputEmbedder, target_embedder: InputEmbedder, positional_encoder: PositionalEncoder, encoder: Encoder, decoder: Decoder, projection: ProjectionLayer) -> None:
        """
        Constructor:
            Initializes the Transformer module. This module is used to encode and decode the input tensor.
        Arguments:
            input_embedder: InputEmbedder - The input embedder to embed the input tensor.
            positional_encoder: PositionalEncoder - The positional encoder to encode the input tensor.
            encoder: Encoder - The encoder to encode the input tensor.
            decoder: Decoder - The decoder to decode the input tensor.
            linear_layer: torch.nn.Linear - The linear layer to apply to the output tensor.
        Outputs:
            - The output tensor from the decoder.
            -- Takes in source and target tensors in R and returns an output tensor in R.
            -- Takes in source and target tensors of shape (batch_size, sequence_length) and returns an output tensor of shape (batch_size, sequence_length, dimensions).
            -- The final linear layer maps the output tensor to the vocabulary size which is a probability distribution over the vocabulary. This helps us to predict the next word in the sequence.
        """
        super(Transformer, self).__init__()
        self.input_embedder = input_embedder
        self.target_embedder = target_embedder
        self.positional_encoder = positional_encoder
        self.encoder = encoder
        self.decoder = decoder
        self.projection = projection

    def encode(self, input_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Encode:
            - Embeds the input tensor.
            - Encodes the input tensor.
            -- Takes in an input tensor in R and returns an output tensor in R^d.
            -- Takes in an input tensor of shape (batch_size, sequence_length) and returns an output tensor of shape (batch_size, sequence_length, dimensions).
        Arguments:
            input_tensor: torch.Tensor - The input tensor to encode.
            mask: torch.Tensor - The mask tensor to apply to the attention scores.
        """
        return self.encoder(self.positional_encoder(self.input_embedder(input_tensor)), mask)
    
    def decode(self, encoder_output: torch.Tensor, encoder_mask: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """
        Decode:
            - Decodes the input tensor.
            -- Takes in an input tensor in R and returns an output tensor in R^d.
            -- Takes in an input tensor of shape (batch_size, sequence_length) and returns an output tensor of shape (batch_size, sequence_length, dimensions).
        Arguments:
            input_tensor: torch.Tensor - The input tensor to decode.
            encoder_output: torch.Tensor - The output tensor from the encoder.
            source_mask: torch.Tensor - The mask tensor to apply to the attention scores for the source tensor.
            target_mask: torch.Tensor - The mask tensor to apply to the attention scores for the target tensor.
        """
        return self.decoder(self.positional_encoder(self.target_embedder(target)), encoder_output, encoder_mask, target_mask)
    
    def project(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Project:
            - Projects the input tensor into a higher dimension.
            -- Takes in an input tensor in R^d and returns an output tensor in R^v.
            -- Takes in an input tensor of shape (batch_size, sequence_length, dimensions) and returns an output tensor of shape (batch_size, sequence_length, vocabulary_size).
        Arguments:
            input_tensor: torch.Tensor - The input tensor to project.
        """
        return self.projection(input_tensor)


def build_transformer(input_vocabulary_size: int, target_vocabulary_size: int, input_sequence_length: int, target_sequence_length: int, dimensions: int, num_layers: int, num_heads: int, dropout: int, linear_projection_dimensions: int) -> Transformer:
    """
    BuildTransfomer:
        - Builds the transformer model.
        -- Takes in the input and target vocabulary sizes, input and target sequence lengths, dimensions, number of layers, number of heads, dropout, and linear projection dimensions.
        -- Returns a transformer model.
    Arguments:
        input_vocabulary_size: int - The number of unique tokens in the input.
        target_vocabulary_size: int - The number of unique tokens in the target.
        input_sequence_length: int - The maximum length of the input sequence.
        target_sequence_length: int - The maximum length of the target sequence.
        dimensions: int - The number of dimensions to embed the input tokens into.
        num_layers: int - The number of layers to apply to the input tensor.
        num_heads: int - The number of heads to split the input tensor into.
        dropout: int - The dropout rate to apply to the output tensor.
        linear_projection_dimensions: int - The number of dimensions to project the input tensor into.
    """
    input_embedder = InputEmbedder(dimensions, input_vocabulary_size)
    target_embedder = InputEmbedder(dimensions, target_vocabulary_size)

    positional_encoder = PositionalEncoder(dimensions, input_sequence_length, dropout)

    encoder_blocks = []
    for _ in range(num_layers):
        self_attention_layer = MultiHeadAttentionLayer(dimensions, num_heads, dropout)
        feed_forward_layer = FeedForwardLayer(dimensions, linear_projection_dimensions, dropout)
        encoder_blocks.append(EncoderBlock(dimensions, self_attention_layer, feed_forward_layer, dropout))

    decoder_blocks = []
    for _ in range(num_layers):
        self_attention_layer = MultiHeadAttentionLayer(dimensions, num_heads, dropout)
        cross_attention_layer = MultiHeadAttentionLayer(dimensions, num_heads, dropout)
        feed_forward_layer = FeedForwardLayer(dimensions, linear_projection_dimensions, dropout)
        decoder_blocks.append(DecoderBlock(dimensions, self_attention_layer, cross_attention_layer, feed_forward_layer, dropout))

    encoder = Encoder(dimensions, torch.nn.ModuleList(encoder_blocks))
    decoder = Decoder(dimensions, torch.nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(dimensions, target_vocabulary_size)

    transformer = Transformer(
        input_embedder=input_embedder, 
        target_embedder=target_embedder, 
        positional_encoder=positional_encoder, 
        encoder=encoder, 
        decoder=decoder, 
        projection=projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    return transformer


if __name__ == '__main__':
    pass