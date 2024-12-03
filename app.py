from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import pipeline
import torch # pip install transformers[torch]
import sys

# Load IMDb dataset
dataset = load_dataset("imdb")

# print(dataset)
# print(dataset['train'][0])  # First training sample

# Use the pre-defined splits
train_data = dataset["train"].shuffle(seed=42).select(range(1000))  
test_data = dataset["test"].shuffle(seed=42).select(range(200))    # Validation subset


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding=True, max_length=128)

train_data = train_data.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)


train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])



# Load the model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)  # 2 labels: positive/negative

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)


# Init Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)

# Train & save the model locally
trainer.train()
results = trainer.evaluate()

trainer.save_model("./results")  
tokenizer.save_pretrained("./results") 


print(results)

# Load the fine-tuned model into a pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="./results")

while True:
    text =  input("Type text for sentiment Analysis...(Type Q to quit):\n")
    if text.lower() == 'q':
        sys.exit()
    print(sentiment_pipeline(text))

    # Get sentiment prediction
    prediction = sentiment_pipeline(text)[0]
    
    label = prediction['label']
    score = prediction['score']
    label_mapping = {
        "LABEL_0": "Negative",
        "LABEL_1": "Positive"
    }
    
    readable_label = label_mapping.get(label, label)  # Default to label if mapping not found
    
    print(f"\nSentiment: {readable_label}")
    print(f"Confidence: {score:.2%}\n")
