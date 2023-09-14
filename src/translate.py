import argparse
import torch
from model import TransformerModel  # Import your Transformer model implementation
from dataset import TranslationDataset  # Import your TranslationDataset class

def translate_sentence(model, sentence, max_length, device):
    # Tokenize the input sentence
    tokens = sentence.split()

    # Convert tokens to tensor
    input_tensor = torch.tensor([source_vocab[token] for token in tokens], dtype=torch.long, device=device).unsqueeze(0)

    # Initialize the target input with the <SOS> token
    target_input = torch.tensor([target_vocab['<SOS>']], dtype=torch.long, device=device).unsqueeze(0)

    # Initialize the translation result
    translation = []

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        for _ in range(max_length):
            # Generate the prediction
            output = model(input_tensor, target_input)

            # Get the most likely next token
            top_value, top_index = output[:, -1, :].topk(1)
            predicted_token = target_vocab.index2word[top_index.item()]

            # Append the predicted token to the translation
            translation.append(predicted_token)

            # If the end of sentence token is predicted, stop generating
            if predicted_token == '<EOS>':
                break

            # Prepare the predicted token for the next iteration
            target_input = torch.tensor([target_vocab[predicted_token]], dtype=torch.long, device=device).unsqueeze(0)

    return ' '.join(translation)

def main():
    parser = argparse.ArgumentParser(description='Translate English text to Hindi')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input English sentence to translate')
    parser.add_argument('--max_length', type=int, default=50, help='Maximum length of the generated translation')
    args = parser.parse_args()

    # Load the trained model
    model = TransformerModel(...)  # Initialize your Transformer model with the same hyperparameters as used during training
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.to(device)

    # Translate the input sentence
    translation = translate_sentence(model, args.input, args.max_length, device)

    print(f'Input: {args.input}')
    print(f'Translation: {translation}')

if __name__ == '__main__':
    main()
