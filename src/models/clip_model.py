import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

class CLIPPredictor:
    def __init__(self):
        self.model_id = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(self.model_id)
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.class_names = ["bus", "sedan", "suv", "truck"]
        
        # Setup few-shot examples using relative paths
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        train_dir = os.path.join(base_dir, "data", "cars.v1i.coco", "train")
        crops_dir = os.path.join(base_dir, "data", "cars.v1i.coco", "crops")
        
        self.examples = [
            (os.path.join(train_dir, "Screenshot-from-2026-02-12-15-18-28_png.rf.28fc4b45cbb7cc0375a0a1d013da7acb.jpg"), "SUV"),
            (os.path.join(train_dir, "Screenshot-from-2026-02-12-14-51-53_png.rf.a420675aec9d267a4918e4f6b624b938.jpg"), "SUV"),
            (os.path.join(train_dir, "Screenshot-from-2026-02-12-14-49-01_png.rf.0a93df9f0be46f0eb90eac2282fc5347.jpg"), "SUV"),
            (os.path.join(crops_dir, "DJI-00760-00001-1-_jpg.rf.ccecf9a0ecc8984c01aaa5fa25516594_ann10.jpg"), "SUV"),
            (os.path.join(train_dir, "Screenshot-from-2026-02-12-15-17-59_png.rf.f83a7cd8ff5091d85bcd518ba63638ef.jpg"), "sedan"),
            (os.path.join(train_dir, "Screenshot-from-2026-02-12-14-44-24_png.rf.9ecb63672a6e06a87bf5875b063d626d.jpg"), "sedan"),
            (os.path.join(train_dir, "Screenshot-from-2026-02-12-15-02-04_png.rf.e778e47b1e388c7441c2fae1c630ec89.jpg"), "sedan"),
            (os.path.join(crops_dir, "DJI_0005-0071_jpg.rf.85e28363e268a259ced325858f5ea023_ann5.jpg"), "sedan"),
            (os.path.join(train_dir, "Screenshot-from-2026-02-12-15-05-15_png.rf.863275a545cedd225d0060f8e92e7fc7.jpg"), "truck"),
            (os.path.join(train_dir, "Screenshot-from-2026-02-12-15-04-52_png.rf.0d6db9bde6f4d40692ce4001d5010966.jpg"), "truck"),
            (os.path.join(train_dir, "Screenshot-from-2026-02-12-14-55-05_png.rf.da96e0c9ae878062bf8cf23474468091.jpg"), "truck"),
            (os.path.join(train_dir, "Screenshot-from-2026-02-12-14-54-08_png.rf.1fcb78dec3616ec5983678d61ed74729.jpg"), "truck"),
            (os.path.join(train_dir, "Screenshot-from-2026-02-12-15-17-07_png.rf.710f4176b646d40b2b99c564172586ca.jpg"), "bus"),
            (os.path.join(crops_dir, "Screenshot-from-2026-02-12-14-59-11_png.rf.b5726dd264543f89310fff9334907344_ann15.jpg"), "bus"),
            (os.path.join(crops_dir, "Screenshot-from-2026-02-12-14-58-26_png.rf.eca11b21c6c6bfd088b7970f4a99e3ac_ann18.jpg"), "bus"),
            (os.path.join(crops_dir, "DJI_0762-00031_jpg.rf.abb1115b71b8b0ff871968c8daf083ea_ann14.jpg"), "bus"),
        ]
        
        self.ref_matrix, self.ref_labels = self.build_memory_bank(self.examples)

    def get_fused_embedding(self, image: Image.Image, label: str):
        image = image.convert("RGB")
        clean_label = label.lower()
        master_prompt = f"An overhead satellite, drone, or aerial view of a {clean_label} on the ground."
        inputs = self.processor(text=[master_prompt], images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            img_features = outputs.image_embeds
            txt_features = outputs.text_embeds
            img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
            txt_features = txt_features / txt_features.norm(p=2, dim=-1, keepdim=True)
            fused_embedding = (img_features + txt_features) / 2.0
            fused_embedding = fused_embedding / fused_embedding.norm(p=2, dim=-1, keepdim=True)
        return fused_embedding

    def build_memory_bank(self, context_data):
        reference_embeddings = []
        reference_labels = []
        for img_path, label in context_data:
            if os.path.exists(img_path):
                img = Image.open(img_path)
                clean_label = label.lower()
                emb = self.get_fused_embedding(img, clean_label)
                reference_embeddings.append(emb)
                reference_labels.append(clean_label)
        
        if reference_embeddings:
            reference_matrix = torch.cat(reference_embeddings)
            return reference_matrix, reference_labels
        return None, None

    def predict_with_text_prompts(self, image: Image.Image):
        image = image.convert("RGB")
        prompts = [f"An overhead satellite, drone, or aerial view of a {cls} parked on the ground." for cls in self.class_names]
        inputs = self.processor(text=prompts, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            img_emb = outputs.image_embeds
            txt_emb = outputs.text_embeds
            img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
            sims = torch.matmul(img_emb, txt_emb.T)
            best_idx = sims.argmax().item()
        return self.class_names[best_idx]

    def predict_new_image(self, image: Image.Image):
        if self.ref_matrix is None:
            return None
        generic_label = "vehicle"
        emb = self.get_fused_embedding(image, generic_label)
        similarities = torch.matmul(emb, self.ref_matrix.T)
        best_match_index = similarities.argmax().item()
        predicted_label = self.ref_labels[best_match_index]
        return predicted_label

    def predict(self, image: Image.Image):
        # Exact logic from predict_combined in notebook
        pred_text = self.predict_with_text_prompts(image)
        
        if self.ref_matrix is not None:
            pred_img = self.predict_new_image(image)
            if pred_text.lower() == pred_img.lower():
                return pred_text.lower()
            else:
                return pred_text.lower()
        else:
            return pred_text.lower()
