
import os
import torch
import argparse
from transformers import MarianMTModel, MarianTokenizer, AdamW, get_scheduler
from datasets import load_dataset
from utils import get_device, preprocess_data, TranslationDataset

def main(data_path, source_lang, target_lang):
    # Parameters
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    max_length = 128
    batch_size = 16
    epochs = 3
    learning_rate = 5e-5

    # Load Dataset
    dataset = load_dataset('csv', data_files={'train': data_path})

    # Tokenizer
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # Tokenize Data
    tokenized_datasets = dataset.map(lambda x: preprocess_data(tokenizer, x, max_length), batched=True)

    # DataLoader
    train_dataset = TranslationDataset(tokenized_datasets['train'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model
    device = get_device()
    model = MarianMTModel.from_pretrained(model_name)
    model.to(device)

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_loader) * epochs
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Training Function
    def train_epoch(model, data_loader, optimizer, device, scheduler):
        model.train()
        total_loss = 0

        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(data_loader)
        return avg_loss

    # Training Loop
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, lr_scheduler)
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss}')

    # Save Model
    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing translation data')
    parser.add_argument('--source_lang', type=str, required=True, help='Source language code')
    parser.add_argument('--target_lang', type=str, required=True, help='Target language code')
    args = parser.parse_args()
    main(args.data_path, args.source_lang, args.target_lang)
