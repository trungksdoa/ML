import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Set the Hugging Face access token
os.environ["HUGGINGFACE_TOKEN"] = "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"
token = os.environ["HUGGINGFACE_TOKEN"]

# Cấu hình BitsAndBytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Model Loading
def load_model(model_name="meta-llama/Meta-Llama-3-8B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=bnb_config,
        device_map="auto",
        token=token
    )
    return model, tokenizer

# Data Processing
def extract_features(data):
    return f"Age: {data['age_months']} months, Length: {data['length_cm']} cm, Weight: {data['weight_g']} g, " \
           f"Water Temp: {data['water_temp']}°C, pH: {data['ph']}, Ammonia: {data['ammonia']} ppm, " \
           f"Nitrite: {data['nitrite']} ppm, Activity: {data['activity_level']}"

def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def prepare_dataset(df):
    def process_example(example):
        features = f"Age: {example['age_months']} months, Length: {example['length_cm']} cm, Weight: {example['weight_g']} g, " \
                   f"Water Temp: {example['water_temp']}°C, pH: {example['ph']}, Ammonia: {example['ammonia']} ppm, " \
                   f"Nitrite: {example['nitrite']} ppm, Activity: {example['activity_level']}"
        label = example['current_health_status']
        return {
            "features": features,
            "label": label
        }
    
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(process_example)
    return dataset

# Model Training
def train_model(model, tokenizer, train_dataset, eval_dataset, output_dir):
    def tokenize_function(examples):
        return tokenizer(examples["features"], padding="max_length", truncation=True, max_length=128)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        # Add the following line to use the accelerate library
        report_to="none",  # Disable reporting to avoid issues with accelerate
    )

    # Ensure the model is not moved manually
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
    )

    trainer.train()
    trainer.save_model(output_dir)

# Prediction
def predict_koi_health(model, tokenizer, data):
    features = extract_features(data)
    prompt = f"Based on the following koi fish data, predict its health status (healthy, at risk, or sick) and provide a probability:\n{features}\nHealth status:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
    
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parse the prediction to extract health status and probability
    # This part may need adjustment based on the model's output format
    if "healthy" in prediction.lower():
        health_status = "healthy"
    elif "at risk" in prediction.lower():
        health_status = "at risk"
    else:
        health_status = "sick"
    
    # Extract probability (this is a placeholder, adjust based on actual output)
    probability = 0.8  # Default probability
    
    return {"health_status": health_status, "probability": probability, "full_prediction": prediction}

# Feedback Incorporation
def incorporate_feedback(model, tokenizer, feedback_data):
    print("Feedback received:", feedback_data)
    print("Model fine-tuning with feedback is not implemented in this example.")
    return {"message": "Feedback logged (fine-tuning not implemented)"}

if __name__ == "__main__":
    try:
        print("Starting koi health predictor...")
        df = load_data_from_csv('/app/data/csv/koi_data_100_samples.csv')
        print(f"Data loaded. Shape: {df.shape}")

        # Prepare dataset
        full_dataset = prepare_dataset(df)
        
        # Use the datasets library's train_test_split
        dataset_dict = full_dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = dataset_dict['train']
        eval_dataset = dataset_dict['test']
        
        model, tokenizer = load_model()
        print("Model loaded successfully.")

        # Train the model
        output_dir = '/app/models/trained_koi_model'
        train_model(model, tokenizer, train_dataset, eval_dataset, output_dir)
        print("Model training completed.")
        
        # Use the trained model for prediction
        sample_data = {
            "age_months": 36,
            "length_cm": 45.0,
            "weight_g": 1500,
            "water_temp": 24.0,
            "ph": 7.5,
            "ammonia": 0.03,
            "nitrite": 0.01,
            "activity_level": "normal",
        }
        
        result = predict_koi_health(model, tokenizer, sample_data)
        print(f"Prediction result: {result}")
        print("Koi health predictor completed successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
