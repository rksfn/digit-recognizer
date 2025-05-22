import torch
import pandas as pd
import numpy as np
from digit_recognizer import DigitRecognizer, DigitDataset
from torch.utils.data import DataLoader

def load_test_data():
    # Load test data
    test_data = pd.read_csv('test.csv')
    X_test = test_data.values.astype('float32') / 255.0  # Normalize pixel values
    
    # Reshape data for CNN (batch_size, channels, height, width)
    X_test = X_test.reshape(-1, 1, 28, 28)
    
    # Create dataset and dataloader
    test_dataset = DigitDataset(torch.FloatTensor(X_test))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return test_loader

def make_predictions():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = DigitRecognizer()
    model.load_state_dict(torch.load('digit_recognizer_model.pth'))
    model = model.to(device)
    model.eval()
    
    # Load test data
    test_loader = load_test_data()
    
    # Make predictions
    predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
    
    # Create submission file
    submission = pd.DataFrame({
        'ImageId': range(1, len(predictions) + 1),
        'Label': predictions
    })
    
    # Save predictions
    submission.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")

if __name__ == '__main__':
    make_predictions() 