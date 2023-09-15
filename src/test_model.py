import torch
from model import TransformerModel  # Import your Transformer model implementation

# Initialize your model and other necessary configurations
# Check if a GPU is available and use it, otherwise use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_vocab_size = 1000
target_vocab_size = 1500
input_seq_length = 120
target_seq_length = 120
d_model = 512
nhead = 8
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward = 2048
dropout = 0.1

def test_model_with_random_data(model, device):
    # Generate random input tensor (English)
    
    input_tensor = torch.randint(0, input_vocab_size, (10, input_seq_length), dtype=torch.long, device=device)

    # Generate random target input tensor (Hindi)
    target_input_tensor = torch.randint(0, target_vocab_size, (10, target_seq_length), dtype=torch.long, device=device)

    # Pass the input tensor through the model for translation
    output = model(input_tensor, target_input_tensor)

    # Process the model output as needed
    # For example, you can decode the output tensor to obtain the translated sentence

    return output

def main():
    model = TransformerModel(input_vocab_size, target_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
    model.to(device)
    print("Device:", device)

    # You can also get more information about the GPU device, if available
    if device.type == "cuda":
        print("GPU Name:", torch.cuda.get_device_name(0))
        print("GPU Memory:", torch.cuda.max_memory_allocated(0) / (1024 * 1024), "MB")
    
    # Call the test function with random data
    output = test_model_with_random_data(model, device)

    # Print or process the model output as needed
    print(output.shape)

if __name__ == '__main__':
    main()
