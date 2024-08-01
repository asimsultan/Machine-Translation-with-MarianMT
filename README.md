
# Machine Translation with MarianMT

Welcome to the Machine Translation with MarianMT project! This project focuses on translating text from one language to another using the MarianMT model.

## Introduction

Machine translation involves converting text from one language to another. In this project, we leverage the power of MarianMT to perform machine translation using a dataset of parallel text.

## Dataset

For this project, we will use a custom dataset of parallel text (source text and target text). You can create your own dataset and place it in the `data/translation_data.csv` file.

## Project Overview

### Prerequisites

- Python 3.6 or higher
- PyTorch
- Hugging Face Transformers
- Datasets
- Pandas

### Installation

To set up the project, follow these steps:

```bash
# Clone this repository and navigate to the project directory:
git clone https://github.com/your-username/marianmt_machine_translation.git
cd marianmt_machine_translation

# Install the required packages:
pip install -r requirements.txt

# Ensure your data includes parallel text (source text and target text). Place these files in the data/ directory.
# The data should be in a CSV file with two columns: source_text and target_text.

# To fine-tune the MarianMT model for machine translation, run the following command:
python scripts/train.py --data_path data/translation_data.csv --source_lang en --target_lang es

# To evaluate the performance of the fine-tuned model, run:
python scripts/evaluate.py --model_path models/ --data_path data/translation_data.csv --source_lang en --target_lang es
```
