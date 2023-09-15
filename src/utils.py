import torch

def load_checkpoint(model, filename):
    """
    Load the model and optimizer state from a checkpoint file.

    Args:
        model (torch.nn.Module): The PyTorch model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        filename (str): The filename of the checkpoint to load.
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def save_checkpoint(model, optimizer, filename):
    """
    Save the model and optimizer state to a checkpoint file.

    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer to be saved.
        filename (str): The filename to save the checkpoint to.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

# Example usage:
# Assuming you have your model and optimizer defined, and you want to save a checkpoint with the filename "my_checkpoint.pth":
# save_checkpoint(model, optimizer, "my_checkpoint.pth")
