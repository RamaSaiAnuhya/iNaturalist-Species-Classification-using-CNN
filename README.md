# iNaturalist-Species-Classification-using-CNNs and Transfer Learning
A Convolutional Neural Network (CNN) to classify images from the iNaturalist dataset using TensorFlow and Keras. 

## Project Overview
This project explores different deep learning approaches for image classification on the iNaturalist 12K dataset, containing images from 10 natural species categories.

The objective was to compare the performance of:
- A Convolutional Neural Network (CNN) built completely from scratch
- Transfer Learning using MobileNetV2
- Fine-Tuning the pretrained MobileNetV2 model

The project demonstrates how transfer learning significantly improves image classification performance compared to training a CNN from scratch on a relatively small dataset.

---

## Dataset
- Dataset: iNaturalist 12K
- Classes: 10
- Task: Multi-class Image Classification
- Framework: TensorFlow / Keras

Classes include:

1. Amphibia
2. Animalia
3. Arachnida
4. Aves
5. Fungi
6. Insecta
7. Mammalia
8. Mollusca
9. Plantae
10. Reptilia

---

## Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

## Project Pipeline
```
Dataset
      │
      ▼
Preprocessing
      │
      ▼
Data Augmentation
      │
      ▼
──────────────────────────────
Approach 1
Custom CNN
──────────────────────────────
      │
      ▼
Evaluation

──────────────────────────────
Approach 2
Transfer Learning
(MobileNetV2)
──────────────────────────────
      │
      ▼
Evaluation

──────────────────────────────
Approach 3
Fine-Tuning
──────────────────────────────
      │
      ▼
Evaluation
```

## Model Architectures
### 1. CNN from Scratch

Features:

- Multiple Convolution Blocks
- Batch Normalization
- ReLU Activation
- Max Pooling
- Global Average Pooling
- Dropout
- L2 Regularization
- Early Stopping
- ReduceLROnPlateau
  
### 2. Transfer Learning

Backbone: MobileNetV2 (ImageNet Pretrained)

Classifier:

- Global Average Pooling
- Dense Layer
- Dropout
- Softmax Output

The pretrained backbone was frozen while training only the classification head.

### 3. Fine-Tuning

The last layers of MobileNetV2 were unfrozen and trained using a very small learning rate to further adapt pretrained features to the iNaturalist dataset.

---
 
## Results 

| Model                           | Test Accuracy |
| ------------------------------- | ------------: |
| CNN from Scratch                |    **40.50%** |
| MobileNetV2 (Transfer Learning) |    **95.06%** |
| MobileNetV2 (Fine-Tuning)       |    **76.34%** |

## Key Learnings
- Training a CNN from scratch on a relatively small dataset is challenging and requires careful architecture design and hyperparameter tuning.
- Batch Normalization, data augmentation, dropout, learning rate scheduling, and regularization improve model stability and generalization.
- Transfer learning dramatically outperforms training from scratch by leveraging features learned from large-scale datasets.
- Fine-tuning pretrained models can further improve performance by adapting high-level representations to the target dataset.

## Future Improvements
- Experiment with EfficientNet and ResNet architectures.
- Hyperparameter optimization using Keras Tuner.
- Model deployment using Streamlit or Gradio.
- Explainability using Grad-CAM.
