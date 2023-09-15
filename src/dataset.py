import torch
from torch.utils.data import Dataset

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing

class TranslationDataset(Dataset):
    def __init__(self, data_path, max_length=20, mode="train"):
        """
        Args:
            data_path (str): Path to the data directory containing train.en.txt and train.hi.txt.
            max_length (int, optional): Maximum sequence length. If specified, sequences longer than this will be truncated.
        """
        self.data_path = data_path
        self.max_length = max_length

        # Load English and Hindi sentences
        if mode=="train":
            self.english_sentences = self.load_sentences(data_path + 'train.en.txt')
            self.hindi_sentences = self.load_sentences(data_path + 'train.hi.txt')
        if mode=="eval":
            self.english_sentences = self.load_sentences(data_path + 'val.en.txt')
            self.hindi_sentences = self.load_sentences(data_path + 'val.hi.txt')

        # Tokenize the sentences
        self.english_sentences, self.hindi_sentences = self.tokenize_sentences(self.english_sentences, self.hindi_sentences)

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        english_sequence = self.english_sentences[idx]
        hindi_sequence = self.hindi_sentences[idx]

        return {
            'input_ids': torch.tensor(english_sequence),
            'target_ids': torch.tensor(hindi_sequence)
        }

    def load_sentences(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            sentences = [line.strip() for line in file.readlines()]
        return sentences

    

    def tokenize_sentences(self, english_sentences, hindi_sentences):
        # Initialize spaCy tokenizers for English and Hindi
        en_tokenizer = Tokenizer.from_file("tokenizer-en.json")
        hi_tokenizer = Tokenizer.from_file("tokenizer-hi.json")  # Replace with an appropriate Hindi language model if available
        # en_tokenizer.post_processor = TemplateProcessing(
        #     single="[SOS] $A [EOS]",
        #     special_tokens=[
        #         ("[SOS]", en_tokenizer.token_to_id("[SOS]")),
        #         ("[EOS]", en_tokenizer.token_to_id("[EOS]")),
        #     ],
        # )

        # hi_tokenizer.post_processor = TemplateProcessing(
        #     single="[SOS] $A [EOS]",
        #     special_tokens=[
        #         ("[SOS]", hi_tokenizer.token_to_id("[SOS]")),
        #         ("[EOS]", hi_tokenizer.token_to_id("[EOS]")),
        #     ],
        # )

        # Tokenize English sentences
        english_tokens = [en_tokenizer.encode(sentence) for sentence in english_sentences]

        # Tokenize Hindi sentences
        hindi_tokens = [hi_tokenizer.encode(sentence) for sentence in hindi_sentences]

        # Extract token texts and handle padding/truncation based on self.max_length
        # Function to pad or truncate sequences to a fixed length
        def pad_or_truncate(sequence, max_length):
            
            if len(sequence) < max_length:
                sequence.extend([en_tokenizer.token_to_id("[PAD]")] * (max_length - len(sequence)))
            elif len(sequence) > max_length:
                sequence = sequence[:max_length]
            
            # Add [SOS] (start of sequence) and [EOS] (end of sequence) tokens
            sequence = [en_tokenizer.token_to_id("[SOS]")] + sequence + [en_tokenizer.token_to_id("[EOS]")]
            return sequence

        english_token_ids = [pad_or_truncate(token.ids, self.max_length) for token in english_tokens]
        hindi_token_ids = [pad_or_truncate(token.ids, self.max_length) for token in hindi_tokens]

    
        return english_token_ids, hindi_token_ids

if __name__ == "__main__":
    # Create an instance of your TranslationDataset
    data_path = "data/"
    max_length = 10  # Set your desired maximum sequence length
    dataset = TranslationDataset(data_path, max_length)
    
    # Choose a specific index (idx) for testing
    idx = 1  # Replace with the index you want to test
    
    # Retrieve data for the chosen index
    sample = dataset[idx]
    
    # Print or process the sample data
    print(f"Input IDs: {sample['input_ids']}")
    print(f"Target IDs: {sample['target_ids']}")

    # from torch.utils.data import DataLoader
    # # Define batch size
    # batch_size = 2  # Set your desired batch size
    # # Create a DataLoader with the custom collate function
    # dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # # Iterate through the DataLoader
    # for batch in dataloader:
    #     input_ids = batch['input_ids']
    #     target_ids = batch['target_ids']
    
    #     # Print or process the batched data as needed
    #     print(f"Batch Input IDs: {input_ids}")
    #     print(f"Batch Target IDs: {target_ids[:-1]}")
