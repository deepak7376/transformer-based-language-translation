import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_vocab_size, target_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()

        # Define your Transformer encoder
        self.encoder = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # Define your Transformer decoder
        self.decoder = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
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
        encoder_output = self.encoder(src_embedded)
        decoder_output = self.decoder(tgt_embedded, encoder_output)

        return self.fc(decoder_output)

# Example of model initialization:
# model = TransformerModel(input_vocab_size, target_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
