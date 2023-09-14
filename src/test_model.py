import torch
from model import TransformerModel  # Import your Transformer model implementation

def test_model_with_random_data(model, device):
    # Generate random input tensor (English)
    input_tensor = torch.randint(0, input_vocab_size, (1, input_seq_length), dtype=torch.long, device=device)

    # Generate random target input tensor (Hindi)
    target_input_tensor = torch.randint(0, target_vocab_size, (1, target_seq_length), dtype=torch.long, device=device)

    # Pass the input tensor through the model for translation
    output = model(input_tensor, target_input_tensor)

    # Process the model output as needed
    # For example, you can decode the output tensor to obtain the translated sentence

    return output

def main():
    # Initialize your model and other necessary configurations
    model = TransformerModel(input_vocab_size, target_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
    model.to(device)
    
    # Call the test function with random data
    output = test_model_with_random_data(model, device)

    # Print or process the model output as needed
    print(output)

if __name__ == '__main__':
    main()
