
import torch
import argparse
from transformers import MarianMTModel, MarianTokenizer
from utils import get_device, preprocess_data, TranslationDataset
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main(model_path, data_path, source_lang, target_lang):
    # Load Model and Tokenizer
    model = MarianMTModel.from_pretrained(model_path)
    tokenizer = MarianTokenizer.from_pretrained(model_path)

    # Device
    device = get_device()
    model.to(device)

    # Load Dataset
    dataset = load_dataset('csv', data_files={'validation': data_path})
    tokenized_datasets = dataset.map(lambda x: preprocess_data(tokenizer, x, max_length=128), batched=True)

    # DataLoader
    eval_dataset = TranslationDataset(tokenized_datasets['validation'])
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=16, shuffle=False)

    # Evaluation Function
    def evaluate(model, data_loader, device):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
                preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return accuracy, precision, recall, f1

    # Evaluate
    accuracy, precision, recall, f1 = evaluate(model, eval_loader, device)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing validation data')
    parser.add_argument('--source_lang', type=str, required=True, help='Source language code')
    parser.add_argument('--target_lang', type=str, required=True, help='Target language code')
    args = parser.parse_args()
    main(args.model_path, args.data_path, args.source_lang, args.target_lang)
