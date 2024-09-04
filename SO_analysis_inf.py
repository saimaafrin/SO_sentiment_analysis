from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load the fine-tuned model and tokenizer
model_path = "/home/safrin/SE/finetuned_SO_emotion_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to predict emotions
def predict_emotions(texts):
    # Tokenize the input texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(outputs.logits).cpu().numpy()
    
    return probs

# Function to interpret results
def interpret_results(probs):
    emotions = ['Anger', 'Fear', 'Joy', 'Love', 'Sadness', 'Surprise']
    results = []
    for prob in probs:
        result = {}
        for emotion, score in zip(emotions, prob):
            result[emotion] = 'Yes' if score > 0.5 else 'No'
        results.append(result)
    return results

# Example usage
texts = [
    "I'M SORRY!!!! I just couldn't help myself.....!",
    "I think first approach is more efficient.",
    "it is hard to tell what might be going wrong where — the problem is likely to be in the code you've not shown.",
    "If you follow this approach, the error will get resolved.",
    "Voila, you just succeed to finetune a model"
]

# Get predictions
predictions = predict_emotions(texts)

# Interpret results
interpreted_results = interpret_results(predictions)

# Print results
for text, result in zip(texts, interpreted_results):
    print(f"Text: {text}")
    print("Emotions detected:")
    for emotion, presence in result.items():
        print(f"  {emotion}: {presence}")
    print()