import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_vocab_size, target_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()

        # Define your Transformer encoder
        self.transformer_model = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.input_embedding = nn.Embedding(input_vocab_size, d_model)
        self.target_embedding = nn.Embedding(target_vocab_size, d_model)
        self.fc = nn.Linear(d_model, target_vocab_size)

    def forward(self, src, tgt):
        src_embedded = self.input_embedding(src)
        tgt_embedded = self.target_embedding(tgt)

        # Pass the source and target sequences through the encoder and decoder
        output = self.transformer_model(src_embedded, tgt_embedded)
        return self.fc(output)

if __name__ == "__main__":

  # Define the model parameters
  num_encoder_layers = 2
  num_decoder_layers = 2
  d_model = 512
  num_heads = 8
  d_ff = 2048
  src_vocab_size = 1000  # Example source vocabulary size
  tgt_vocab_size = 5000  # Example target vocabulary size
  dropout = 0.1
  seq_len = 100

  # Set the parameters
  batch_size = 1

  # Example of model initialization:
  model = TransformerModel(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff, dropout)

  # Create random dummy input data within the vocab_size range
  src_input = torch.randint(0, src_vocab_size, (batch_size, seq_len))
  tgt_input = torch.randint(0, tgt_vocab_size, (batch_size, seq_len))

  # Forward pass through the Transformer
  output_logits = model(src_input, tgt_input[:-1])
  print("Transformer Output", output_logits.shape)
