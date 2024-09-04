# This script also works But it was a test demo script
 
# Import required modules
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import os

# Set environment variable to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set the path to your dataset
excel_file_path = 'Emotions_GoldSandard_andAnnotation.xlsx'
xls = pd.ExcelFile(excel_file_path)
sheet_names = xls.sheet_names

# Load and preprocess the data
def load_and_label(sheet_name):
    data = pd.read_excel(xls, sheet_name=sheet_name)
    data['Emotion'] = data['Gold Label'].apply(lambda x: sheet_name.split('_')[0] if pd.notna(x) else None)
    return data[['Text', 'Emotion']]

emotion_sheets = [load_and_label(name) for name in sheet_names]
all_emotions_data = pd.concat(emotion_sheets, ignore_index=True)

# Pivot to create multi-label format
all_emotions_data['Emotion_present'] = 1
pivot_data = all_emotions_data.pivot_table(index='Text', columns='Emotion', values='Emotion_present', fill_value=0).reset_index()

# Split the data
train_data, test_data = train_test_split(pivot_data, test_size=0.2, random_state=42)

# Convert to Hugging Face's Dataset format
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

# Initialize tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the pre-trained model configuration
config = AutoConfig.from_pretrained(model_name, num_labels=6, problem_type="multi_label_classification")

# Load the pre-trained model with our custom configuration
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

def tokenize_and_format(examples):
    tokenized_inputs = tokenizer(examples['Text'], padding="max_length", truncation=True, max_length=512)
    
    labels = [[float(examples[label][i]) for label in ['Anger', 'Fear', 'Joy', 'Love', 'Sadness', 'Surprise']] 
              for i in range(len(examples['Text']))]
    
    tokenized_inputs['labels'] = labels
    
    return tokenized_inputs

# Apply this function to the dataset
train_dataset = train_dataset.map(tokenize_and_format, batched=True)
test_dataset = test_dataset.map(tokenize_and_format, batched=True)

# Set format for PyTorch
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
)

# Function to compute metrics for multi-label classification
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (np.array(logits) > 0).astype(int)
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='micro')
    recall = recall_score(labels, predictions, average='micro')
    f1 = f1_score(labels, predictions, average='micro')
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)