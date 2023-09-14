import torch
from torch.utils.data import Dataset
import spacy

class TranslationDataset(Dataset):
    def __init__(self, data_path, max_length=None):
        """
        Args:
            data_path (str): Path to the data directory containing train.en.txt and train.hi.txt.
            max_length (int, optional): Maximum sequence length. If specified, sequences longer than this will be truncated.
        """
        self.data_path = data_path
        self.max_length = max_length

        # Load English and Hindi sentences
        self.english_sentences = self.load_sentences(data_path + 'train.en.txt')
        self.hindi_sentences = self.load_sentences(data_path + 'train.hi.txt')

        # Tokenize the sentences
        self.english_sentences, self.hindi_sentences = self.tokenize_sentences(self.english_sentences, self.hindi_sentences)

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        english_sequence = self.english_sentences[idx]
        hindi_sequence = self.hindi_sentences[idx]

        return {
            'input_text': english_sequence,
            'target_text': hindi_sequence
        }

    def load_sentences(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            sentences = [line.strip() for line in file.readlines()]
        return sentences

    def tokenize_sentences(self, english_sentences, hindi_sentences):
        # Initialize spaCy tokenizers for English and Hindi
        en_tokenizer = spacy.load("en_core_web_sm")
        hi_tokenizer = spacy.load("xx_ent_wiki_sm")  # Replace with an appropriate Hindi language model if available

        # Tokenize English sentences
        english_tokens = [en_tokenizer(sentence) for sentence in english_sentences]

        # Tokenize Hindi sentences
        hindi_tokens = [hi_tokenizer(sentence) for sentence in hindi_sentences]

        # Extract token texts and handle padding/truncation based on self.max_length
        if self.max_length is not None:
            english_tokens = [token[:self.max_length] for token in english_tokens]
            hindi_tokens = [token[:self.max_length] for token in hindi_tokens]

        # Convert tokenized sentences to lists of token texts
        english_token_texts = [[token.text for token in tokens] for tokens in english_tokens]
        hindi_token_texts = [[token.text for token in tokens] for tokens in hindi_tokens]

        return english_token_texts, hindi_token_texts
