# Custom Model with Skip Connections and Depth Scaling DL/

## 📌 Overview
This project implements and compares deep neural networks with residual connections for image classification, focusing on two challenging datasets: **Facial Expression Recognition (FER)** and **Butterfly Species Classification**. Our custom CNN architecture combines principles from VGGNet and ResNet to address vanishing gradients while maintaining computational efficiency.

## ✨ Key Features
- **Custom CNN Model** with depth scaling (32-512 filters) and skip connections
- **State-of-the-Art Models** implementation:
  - VGG19 with/without fine-tuning
  - ResNet50 with/without fine-tuning
- **Advanced Training Techniques**:
  - Dynamic learning rate scheduling
  - Strategic layer unfreezing for fine-tuning
  - 30% dropout regularization
- **Multi-Dataset Evaluation**:
  - 100-class Butterfly Species (12,594 images)
  - 7-class Facial Expressions (28,709 images)

## 🗂️ Dataset Overview

### 🦋 Butterfly Species Dataset
- **100 classes** with 100-200 images per species
- **Input size**: 224x224x3
- **Split**:
  - Train: 12,594 images
  - Validation: 500 images
  - Test: 500 images

### 😃 Facial Expression Recognition (FER)
- **7 emotion classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Input size**: 48x48x1 (grayscale)
- **Split**:
  - Train: 28,709 images
  - Test: 3,589 images

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/image-classification-resnets.git
cd image-classification-resnets

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Training Custom Model
```python
from models.custom_model import build_custom_cnn

model = build_custom_cnn(input_shape=(224,224,3), num_classes=100)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=20, validation_data=val_generator)
```

### Fine-tuning Pre-trained Models
```python
from transfer_learning import fine_tune_vgg19

ft_model = fine_tune_vgg19(
    base_model_weights='imagenet',
    num_classes=7,
    unfreeze_layers=12
)
ft_model.fit(fer_train_generator, epochs=10)
```

## 📊 Results

### Performance Comparison
| Model                  | Butterfly Accuracy | FER Accuracy | Parameters |
|------------------------|--------------------|--------------|------------|
| Custom CNN             | 88.82%             | 61.24%       | 9.04M      |
| ResNet50 (Fine-tuned)  | 84.80%             | 47.20%       | 18.29M     |
| VGG19 (Fine-tuned)     | 79.80%             | 67.34%       | 18.32M     |

### Key Findings
- 🏆 **Custom Model Superiority**: Achieved best performance on Butterfly dataset (88.82%) with 60% fewer parameters than fine-tuned models
- ⚡ **Training Efficiency**: 195s/epoch vs 190s for fine-tuned models (T4 GPU)
- 🧠 **Transfer Learning Impact**: Fine-tuning improved VGG19 performance by 55% on FER dataset

## 🗂️ Project Structure
```
image-classification-resnets/
├── data/
│   ├── butterfly/           # Raw butterfly images
│   └── fer/                 # Facial expression dataset
├── models/
│   ├── custom_model.py      # Custom CNN implementation
│   ├── vgg19.py             # VGG19 modifications
│   └── resnet50.py          # ResNet50 implementation
├── notebooks/
│   └── performance_analysis.ipynb  # Result visualizations
├── utils/
│   ├── data_loader.py       # Dataset preprocessing
│   └── callbacks.py         # Custom training callbacks
├── requirements.txt
└── README.md
```

## 🧠 Model Architecture Highlights

### Custom CNN Design
```python
def residual_block(x, filters, kernel_size=3):
    shortcut = x
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])  # Skip connection
    return ReLU()(x)
```

## 📈 Training Dynamics
- **Optimizer**: Adam with initial LR=0.0001
- **LR Schedule**: Reduce by 0.2 factor on plateau
- **Early Stopping**: Patience=5 epochs
- **Batch Size**: 32 (optimized for GPU memory)

## 🚧 Challenges & Solutions
1. **Vanishing Gradients**  
   ⇨ Implemented residual connections with identity mapping

2. **Class Imbalance (FER)**  
   ⇨ Applied weighted class sampling and focal loss

3. **Overfitting**  
   ⇨ 30% dropout + L2 regularization (λ=0.0001)
