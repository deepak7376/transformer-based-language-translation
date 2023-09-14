import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TranslationDataset  # You'll need to create a dataset module
from model import TransformerModel  # Your Transformer model implementation

def evaluate(model, eval_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt in eval_loader:
            src, tgt = src.to(device), tgt.to(device)

            output = model(src, tgt[:-1])  # Exclude the last token from target

            loss = criterion(output.view(-1, output.size(-1)), tgt[1:].view(-1))
            total_loss += loss.item()

    return total_loss / len(eval_loader)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Transformer-based Language Translation Evaluator')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda or cpu)')
    args = parser.parse_args()

    # Create a DataLoader for your evaluation dataset
    eval_dataset = TranslationDataset(args.data_path, mode='eval')
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    # Load the model and set it to evaluation mode
    model = TransformerModel(...)  # Initialize your Transformer model with the same hyperparameters as used during training
    model.to(args.device)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    eval_loss = evaluate(model, eval_loader, criterion, args.device)
    print(f'Evaluation Loss: {eval_loss:.4f}')

if __name__ == '__main__':
    main()
