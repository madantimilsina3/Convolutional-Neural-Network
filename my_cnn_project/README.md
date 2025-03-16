# Cats vs. Dogs Image Classification with Deep Learning

## Project Overview 🚀
**Objective**: Build a convolutional neural network (CNN) to classify images of cats and dogs using transfer learning and GPU acceleration.  

**Key Outcomes**:  
- Achieved **95.56% validation accuracy** in 10 epochs using MobileNetV2  
- Implemented data augmentation to reduce overfitting  
- Leveraged NVIDIA CUDA on WSL2 for 5x faster training  

**Demo**:  
Website URL

---

## Technical Implementation 🔧

### Dataset  
- Source: [Kaggle Cats vs. Dogs](https://www.kaggle.com/c/dogs-vs-cats)  
- **25,000 images** split into:  
  - Training: 20,000 images (80%)  
  - Validation: 5,000 images (20%)  

### Model Architecture  
```python
model = models.Sequential([
    MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3)),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])