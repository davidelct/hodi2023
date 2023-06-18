# necessary imports
import re
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

# load and preprocess the dataset
def preprocess_text(text):
    text = re.sub(r"@\w+", "", text)
    text = text.replace("#", "")
    text = text.strip()
    return text

train_data = pd.read_csv('./HODI_2023_train_subtaskA.tsv', sep='\t')
train_data["text"] = train_data["text"].apply(preprocess_text)

test_data = pd.read_csv('./HODI_2023_test_subtaskA.tsv', sep='\t')
test_data["text"] = test_data["text"].apply(preprocess_text)

class HomotransphobiaDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __getitem__(self, idx):
        item = {}
        item["input_ids"] = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=78)["input_ids"]
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)
      
      
tokenizer = AutoTokenizer.from_pretrained('Musixmatch/umberto-commoncrawl-cased-v1')
train_dataset = HomotransphobiaDataset(train_data["text"], train_data['homotransphobic'], tokenizer)

# load and finetune the model
model = AutoModelForSequenceClassification.from_pretrained('Musixmatch/umberto-commoncrawl-cased-v1', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    save_strategy='epoch',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
)

trainer.train()

# generate and save predictions
test_dataset = HomotransphobiaDataset(test_data["text"], [0]*len(test_data), tokenizer)

predictions = trainer.predict(test_dataset)
probs = np.exp(predictions.predictions) / np.exp(predictions.predictions).sum(-1, keepdims=True)
preds = np.argmax(predictions.predictions, axis=-1)

test_data['predicted_homotransphobia'] = preds
test_data.to_csv('metzi.A.run3.tsv', sep='\t', index=False)
