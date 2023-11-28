# Language Translation using Transformer

This project demonstrates a language translation system that translates English text to Hindi using a Transformer-based neural network. The Transformer architecture has proven to be highly effective for sequence-to-sequence tasks like machine translation, making it a powerful choice for this project.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Model Checkpoints](#model-checkpoints)
- [Contributing](#contributing)
- [License](#license)

## Overview

The project's main components include:

- **Transformer Model**: A state-of-the-art Transformer model for sequence-to-sequence tasks.

- **Data Processing**: Scripts for data preprocessing and tokenization.

- **Training**: Code for training the Transformer model on your dataset.

- **Evaluation**: Evaluation scripts to measure the model's translation quality.

- **Inference**: Scripts to perform translation on new input data.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/deepak7376/transformer-based-language-translation.git
   cd transformer-based-language-translation
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Dataset

To train the translation model, you will need a dataset containing pairs of English and Hindi sentences. Organize your dataset in a directory like this:

```
data/
│
├── train.en.txt   # English training data
├── train.hi.txt   # Hindi training data
├── dev.en.txt     # English development (validation) data
├── dev.hi.txt     # Hindi development (validation) data
```

### Training

To train the model, use the `train.py` script:

```bash
python train.py --data_path data/ --batch_size 64 --num_epochs 10
```

You can adjust the batch size, number of epochs, and other hyperparameters in the script.

### Evaluation

Evaluate the model using the `eval.py` script:

```bash
python eval.py --data_path data/ --checkpoint_path checkpoints/model_checkpoint.pth
```

The script will calculate evaluation metrics like BLEU score to assess translation quality.

### Inference

For translation on new data, you can use the `translate.py` script:

```bash
python translate.py --checkpoint_path checkpoints/model_checkpoint.pth --input "Enter your English text here"
```

Replace `"Enter your English text here"` with the text you want to translate.

## Model Checkpoints

You can find pre-trained model checkpoints in the `checkpoints/` directory. Feel free to use these checkpoints for inference or as a starting point for further training.

## Contributing

Contributions to this project are welcome! If you'd like to contribute, please follow our [contributing guidelines](CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE).
