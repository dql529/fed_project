import numpy as np
from flask import Flask, request, jsonify
from Model import Net18,data,data_test
import torch
import torch.nn as nn
import pickle
import base64
import requests
import time
import multiprocessing
from time import sleep
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net18().to(device)
data_device  = data.to(device)
data_test_device = data_test.to(device)  
# 训练参数
# 定义损失函数和优化器
num_epochs = 10000

learning_rate = 0.001
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


if device.type == 'cuda':
    print(f"Using device: {device}, GPU name: {torch.cuda.get_device_name(device.index)}")
else:
    print(f"Using device: {device}")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Convert the model's output probabilities to binary predictions
def to_predictions(outputs):
    return (torch.sigmoid(outputs) > 0.5).float()

# Evaluate the model on the test data
def evaluate(data_test_device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Do not calculate gradients to save memory
        outputs_test = model(data_test_device)
        predictions_test = to_predictions(outputs_test)
        # Calculate metrics
        accuracy = accuracy_score(data_test_device.y.cpu(), predictions_test.cpu())
        precision = precision_score(data_test_device.y.cpu(), predictions_test.cpu())
        recall = recall_score(data_test_device.y.cpu(), predictions_test.cpu())
        f1 = f1_score(data_test_device.y.cpu(), predictions_test.cpu())
        return accuracy, precision, recall, f1

# Train the model and evaluate at the end of each epoch
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    outputs = model(data_device)

    loss = criterion(outputs, data_device.y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Evaluate
    accuracy, precision, recall, f1 = evaluate(data_test_device)
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Loss: {loss.item()}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    
