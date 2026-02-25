# Few-Shot Aerial Vehicle Classification – VIT & CLIP 

#This branch explores two approaches for multi-label aerial vehicle classification under a severe few-shot constraint (20 images total):

1. Fine-tuned Vision Transformer (ViT)
2. CLIP image embeddings with Logistic Regression

The goal was to evaluate model behavior, generalization, and stability in extremely low-data settings.



## Dataset Setup
- Total images: 20
- Training set: 16 images
- Test set: 4 images
- Classes: sedan, bus, truck, SUV
- Multi-label setting (an image may contain multiple vehicles)



## Approach 1: Vision Transformer (ViT)

### Model
- Backbone: ViT-Base (patch16, 224)
- Pretrained on ImageNet
- Only the classification head was trained
- Backbone was frozen to reduce overfitting

### Results
- Best test accuracy reached **93%**
- Performance was high on the fixed test split

### Observed Issue
Despite high test accuracy, the model **failed to generalize to new unseen images**.
Predictions on new aerial images were unstable and often incorrect.



## Approach 2: CLIP + Logistic Regression

### Model
- CLIP image encoder 
- Used as a frozen feature extractor
- Logistic Regression (One-vs-Rest) for multi-label classification

### Why CLIP
- CLIP is trained on large-scale image–text data
- Provides strong semantic image embeddings
- Better suited for few-shot learning compared to fine-tuning deep models

### Results
- Slightly lower peak accuracy than ViT




