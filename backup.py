import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

# Model Definition
class KoiHealthPredictor(nn.Module):
    def __init__(self, input_size):
        super(KoiHealthPredictor, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Data Processing
def extract_features(data):
    features = [
        data['age_months'],
        data['length_cm'],
        data['weight_g'],
        data['water_temp'],
        data['ph'],
        data['ammonia'],
        data['nitrite']
    ]
    
    activity_map = {'normal': 0, 'lethargic': 1, 'hyperactive': 2}
    features.append(activity_map.get(data['activity_level'], 0))
    
    return features

def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    
    X = df[['age_months', 'length_cm', 'weight_g', 'water_temp', 'ph', 'ammonia', 'nitrite', 'activity_level']].copy()
    
    activity_map = {'normal': 0, 'lethargic': 1, 'hyperactive': 2}
    X['activity_level'] = X['activity_level'].map(activity_map)
    
    health_map = {'healthy': 0, 'at risk': 1, 'sick': 2}
    y = df['current_health_status'].map(health_map)
    
    return X.values, y.values

# Dataset and DataLoader
class KoiDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Model Training
def train_model(X, y, num_epochs=100, batch_size=32):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)

    train_dataset = KoiDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_size = X_train.shape[1]
    model = KoiHealthPredictor(input_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        test_loss = criterion(y_pred, y_test_tensor)
        _, predicted = torch.max(y_pred, 1)
        accuracy = (predicted == y_test_tensor).float().mean().item()
        print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

    return model, scaler

# Prediction
def predict_koi_health(model, scaler, data):
    features = extract_features(data)
    features_scaled = scaler.transform([features])
    features_tensor = torch.FloatTensor(features_scaled)
    
    model.eval()
    with torch.no_grad():
        logits = model(features_tensor)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
    
    health_status = ["healthy", "at risk", "sick"][prediction]
    probability = probabilities[0][prediction].item()
    return {"health_status": health_status, "probability": probability}

# Model Saving and Loading
def save_model(model, scaler, model_path, scaler_path):
    torch.save(model.state_dict(), model_path)
    torch.save(scaler, scaler_path)

def load_model(model_path, scaler_path, input_size):
    model = KoiHealthPredictor(input_size)
    state_dict = torch.load(model_path)
    
    if state_dict['layer1.weight'].shape[1] != input_size:
        old_input_size = state_dict['layer1.weight'].shape[1]
        new_weight = torch.zeros(64, input_size)
        new_weight[:, :old_input_size] = state_dict['layer1.weight']
        state_dict['layer1.weight'] = new_weight
    
    model.load_state_dict(state_dict)
    scaler = torch.load(scaler_path)
    return model, scaler

# Feedback Incorporation
def incorporate_feedback(model, scaler, feedback_data, model_path, scaler_path):
    features = extract_features(feedback_data)
    true_label = feedback_data['true_health_status']
    
    health_map = {'healthy': 0, 'at risk': 1, 'sick': 2}
    true_label_numeric = health_map[true_label]
    
    features_scaled = scaler.transform([features])
    features_tensor = torch.FloatTensor(features_scaled)
    true_label_tensor = torch.LongTensor([true_label_numeric])
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    optimizer.zero_grad()
    outputs = model(features_tensor)
    loss = criterion(outputs, true_label_tensor)
    loss.backward()
    optimizer.step()
    
    save_model(model, scaler, model_path, scaler_path)
    
    return {"message": "Feedback incorporated successfully"}

# Main Execution
if __name__ == "__main__":
    try:
        print("Starting koi health predictor...")
        X, y = load_data_from_csv('/app/data/csv/koi_data_100_samples.csv')
        print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}")
        
        model, scaler = train_model(X, y)
        print("Model training completed.")
        
        save_model(model, scaler, '/app/models/koi_health_model.pth', '/app/models/scaler.pkl')
        print("Model and scaler saved.")
        
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
        
        result = predict_koi_health(model, scaler, sample_data)
        print(f"Prediction result: {result}")
        print("Koi health predictor completed successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise