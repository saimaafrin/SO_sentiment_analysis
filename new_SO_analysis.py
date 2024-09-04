# Import required modules
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.nn import BCEWithLogitsLoss
import os
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
#print(all_emotions_data)

# Pivot to create multi-label format
all_emotions_data['Emotion_present'] = 1
pivot_data = all_emotions_data.pivot_table(index='Text', columns='Emotion', values='Emotion_present', fill_value=0).reset_index()
#print(pivot_data.iloc[20])

# Split the data
train_data, test_data = train_test_split(pivot_data, test_size=0.2, random_state=42)

# Convert to Hugging Face's Dataset format
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

# Initialize tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6, problem_type="multi_label_classification")

'''def tokenize_and_format(examples):
    # Tokenize the text
    tokenized_inputs = tokenizer(examples['Text'], padding="max_length", truncation=True, max_length=512)
    
    # Extract each label column separately and then combine
    labels = [examples[col] for col in ['Anger', 'Fear', 'Joy', 'Love', 'Sadness', 'Surprise']]
    # Ensure labels are correctly structured as a list of lists
    # This requires adjusting how the labels are combined
    labels = list(map(list, zip(*labels)))
    
    tokenized_inputs['labels'] = labels
    labels = [[int(examples[label][i]) for label in ['Anger', 'Fear', 'Joy', 'Love', 'Sadness', 'Surprise']] for i in range(len(examples['Text']))]
    
    tokenized_inputs['labels'] = labels
    
    return tokenized_inputs'''

def tokenize_and_format(examples):
    tokenized_inputs = tokenizer(examples['Text'], padding="max_length", truncation=True, max_length=512)
    
    labels = torch.tensor([
        [examples[label][i] for label in ['Anger', 'Fear', 'Joy', 'Love', 'Sadness', 'Surprise']]
        for i in range(len(examples['Text']))
    ], dtype=torch.float)
    
    tokenized_inputs['labels'] = labels
    
    return tokenized_inputs

# Tokenize and prepare dataset
'''def tokenize_and_format(examples):
    # This will apply the tokenizer to the text and also extract labels as a list
    return tokenizer(examples['Text'], padding="max_length", truncation=True, max_length=512), {'labels': examples[['Anger', 'Fear', 'Joy', 'Love', 'Sadness', 'Surprise']].values.tolist()}
'''

# Apply this function to the dataset
train_dataset = train_dataset.map(tokenize_and_format, batched=True)
test_dataset = test_dataset.map(tokenize_and_format, batched=True)

#print(train_dataset[0])

# Set format for PyTorch
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Function to compute metrics for multi-label classification
'''def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted')
    }
'''
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Sigmoid function to convert logits to multi-label probabilities
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    # Threshold probabilities to convert to binary output
    predictions = (probs > 0.5).astype(int)
    
    # Calculate metrics suitable for multi-label classification
    precision = precision_score(labels, predictions, average='micro')
    recall = recall_score(labels, predictions, average='micro')
    f1 = f1_score(labels, predictions, average='micro')
    accuracy = accuracy_score(labels, predictions)  # This is the subset accuracy
    
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics  # Define this function if you need custom metrics
)

# Train the model
trainer.train()

# Save the model
model_save_path = "./finetuned_SO_emotion_model"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Evaluate the model
results = trainer.evaluate()
print(results)
