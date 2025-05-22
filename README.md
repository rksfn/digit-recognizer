# Handwritten Digit Recognizer

A deep learning model that recognizes handwritten digits using PyTorch. This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset.

## Features

- CNN-based digit recognition model
- 98.98% validation accuracy
- Easy-to-use training and prediction scripts
- Support for both CPU and GPU training

## Project Structure

```
digit-recognizer/
├── digit_recognizer.py    # Main training script
├── predict.py            # Script for making predictions
├── requirements.txt      # Python dependencies
├── train.csv            # Training dataset
├── test.csv             # Test dataset
└── README.md            # This file
```

## Requirements

- Python 3.8+
- PyTorch
- pandas
- numpy
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/digit-recognizer.git
cd digit-recognizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model:

```bash
python digit_recognizer.py
```

This will:
- Load and preprocess the training data
- Train the CNN model
- Save the trained model as 'digit_recognizer_model.pth'

### Making Predictions

To make predictions on new data:

```bash
python predict.py
```

This will:
- Load the trained model
- Process the test data
- Generate predictions in 'submission.csv'

## Model Architecture

The model uses a CNN architecture with:
- 2 convolutional layers
- Max pooling
- Dropout for regularization
- Fully connected layers

## Performance

The model achieves:
- 98.98% validation accuracy
- Fast training and inference
- Good generalization on unseen data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 