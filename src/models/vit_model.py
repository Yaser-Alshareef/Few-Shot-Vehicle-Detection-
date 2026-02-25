import torch
import timm
from PIL import Image
from torchvision import transforms
import os

class ViTPredictor:
    def __init__(self, model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_names = ["sedan", "bus", "truck", "suv"]
        self.img_size = 224
        self.thr = 0.5
        
        self.tfm = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])
        
        self.model_loaded = False
        if model_path and os.path.exists(model_path):
            try:
                ckpt = torch.load(model_path, map_location=self.device)
                self.class_names = ckpt.get("class_names", self.class_names)
                self.img_size = ckpt.get("img_size", self.img_size)
                self.thr = ckpt.get("threshold", 0.5)
                model_name = ckpt.get("model_name", "vit_base_patch16_224")
                
                self.model = timm.create_model(model_name, pretrained=False, num_classes=len(self.class_names))
                self.model.load_state_dict(ckpt["state_dict"])
                self.model = self.model.to(self.device)
                self.model.eval()
                self.model_loaded = True
            except Exception as e:
                print(f"Error loading ViT model: {e}")

    def predict(self, image: Image.Image):
        if not self.model_loaded:
            return "ViT Model weights not found. Please train the model first and provide the .pt file."
            
        image = image.convert("RGB")
        x = self.tfm(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        pred_labels = [self.class_names[i] for i, p in enumerate(probs) if p >= self.thr]
        if len(pred_labels) == 0:
            pred_labels = [self.class_names[int(probs.argmax())]]
            
        return ", ".join(pred_labels)
