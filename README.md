# 🚗 Few-Shot Vehicle Detection & Classification

# 📌 Problem Statement

## Build a computer vision system that works with aerial/overhead vehicle imagery under an extreme data constraint:

- Total labeled images allowed: 20
- 4 classes:
  - Sedan
  - SUV
  - Truck
  - Bus
- 5 images per class

The project focuses on designing a robust few-shot learning pipeline that performs well despite severe data limitations.

---

## 🧠 Our Journey & Approach

### Phase 1: Object Detection (YOLO)
Initially, we approached this as an object detection problem. We experimented with various YOLO model sizes and found that **YOLOv26n (Nano)** performed the best given the tiny dataset. 
- **The Problem:** While we achieved a 98% mAP during training, we quickly realized the model was simply **memorizing** the training data. It failed to generalize to unseen aerial images because 20 images are not enough to learn robust bounding box regression and feature extraction simultaneously from scratch.

### Phase 2: Pivot to Classification (Few-Shot & Zero-Shot)
To combat the overfitting issue, we pivoted from Object Detection to **Image Classification**. Instead of training a model from scratch to find bounding boxes, we decided to leverage massive pre-trained foundation models to extract features from cropped vehicle images.

We implemented and compared four different state-of-the-art approaches:

#### 1. CLIP (Contrastive Language-Image Pretraining)
We explored two different ways to utilize CLIP's powerful semantic embeddings:

**A. Zero-Shot & Few-Shot Prompting**
- **How it works:** We used OpenAI's `clip-vit-base-patch32`. We first tried pure Zero-Shot using text prompts ("An overhead satellite view of a [CLASS] parked on the ground"). We then improved it by building a **Fused Memory Bank** that combines both image embeddings and text embeddings from our 20 reference images.
- **Why it helps:** CLIP already understands the semantic relationship between the word "SUV" and the visual features of an SUV, making it incredibly powerful for low-data regimes.

**B. CLIP + Logistic Regression**
- **How it works:** We used the CLIP image encoder as a frozen feature extractor. We then trained a simple Logistic Regression (One-vs-Rest) classifier on top of these extracted features for multi-label classification.
- **Why it helps:** CLIP is trained on large-scale image–text data, providing strong semantic image embeddings that are much better suited for few-shot learning compared to fine-tuning deep models from scratch.
- **Results:** Achieved slightly lower peak accuracy than the ViT model on the test split, but provided a more stable feature space.

#### 2. DINOv2 (Self-Supervised Vision Transformer)
- **Approach:** Few-Shot Geometric Matching.
- **How it works:** We used Meta's `facebook/dinov2-base`. Unlike CLIP, DINOv2 is a pure-vision model trained without text. We extracted the spatial/geometric embeddings of our 20 reference images to build a memory bank. New images are classified based on cosine similarity to the closest geometric match in the memory bank.
- **Why it helps:** DINOv2 excels at understanding object structure and boundaries, which is perfect for overhead imagery where shapes are distinct.

#### 3. ViT (Vision Transformer)
- **Approach:** Transfer Learning / Fine-Tuning.
- **How it works:** We used `timm` to load a pre-trained `vit_base_patch16_224` (pretrained on ImageNet) and fine-tuned only the classification head using our 20 images. The backbone was frozen to reduce overfitting.
- **Results:** Best test accuracy reached 93%. Performance was high on the fixed test split.
- **Observed Issue:** Despite high test accuracy, the model failed to generalize to new unseen images. Predictions on new aerial images were unstable and often incorrect, indicating that even with a frozen backbone, 20 images are not enough to train a robust classification.

#### 4. VLM (Vision-Language Model - Gemini 2.5 Flash)
- **Approach:** In-Context Few-Shot Prompting.
- **How it works:** We passed our 20 reference images directly into the prompt context of Google's Gemini 2.5 Flash, along with their labels, and asked it to classify a new target image based strictly on the visual patterns provided.
- **Why it helps:** VLMs have immense reasoning capabilities and can perform "in-context learning" without any weight updates or training required.

---

## 🚀 Interactive Streamlit App

We built a Streamlit application to easily compare the performance of these four models side-by-side.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Yaser-Alshareef/Few-Shot-Vehicle-Detection -.git
   cd Few-Shot-Vehicle-Detection-
   ```

2. Install the required dependencies:
   ```bash
   pip install streamlit torch torchvision transformers timm google-generativeai python-dotenv pandas pillow scikit-learn
   ```

3. **For the VLM (Gemini) Model:**
   Create a `.env` file in the root directory and add your API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```


### Running the App

Run the following command in your terminal:
```bash
python -m streamlit run app.py
```

## 📁 Project Structure
```text
├── app.py                  # Main Streamlit application
├── data/                   # Dataset 
│   ├── crops/              # Cropped vehicle images
│   └── train/              # Full training images
├── notebooks/              # Jupyter notebooks for experimentation
│   ├── CLIP.ipynb
│   ├── DINO.ipynb
│   ├── VIT.ipynb
│   └── VLM.ipynb
├── src/
│   ├── models/             # Extracted model inference logic
│   │   ├── clip_model.py
│   │   ├── dino_model.py
│   │   ├── vit_model.py
│   │   └── vlm_model.py
└── test/                   # Test images
```

## 📊 Evaluation & Metrics
Because we pivoted to classification, our primary metrics are **Accuracy**, **Macro F1-Score**, and the **Confusion Matrix**. 
By using foundation models (CLIP, DINOv2, VLM) as feature extractors/reasoners rather than training from scratch, we successfully bypassed the memorization issues we faced with YOLO, achieving robust generalization on unseen aerial images despite the 20-image constraint.
