
import torch
from torch.utils.data import Dataset
from transformers import MarianTokenizer

class TranslationDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item['input_ids'], item['attention_mask'], item['labels']

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_data(tokenizer, examples, max_length):
    tokenized_inputs = tokenizer(
        examples['source_text'],
        truncation=True,
        padding='max_length',
        max_length=max_length
    )
    tokenized_inputs["labels"] = tokenizer(
        examples['target_text'],
        truncation=True,
        padding='max_length',
        max_length=max_length
    )["input_ids"]
    return tokenized_inputs