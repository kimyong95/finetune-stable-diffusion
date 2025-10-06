import torch
from transformers import CLIPModel, CLIPProcessor
from torch import nn
from huggingface_hub import hf_hub_download
from PIL import Image
from typing import List, Optional, Union
import numpy as np
from paddleocr import PaddleOCR
from Levenshtein import distance
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)

class BaseReward(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def __call__(self, images: List[Image.Image], prompts: Optional[List[str]] = None) -> torch.Tensor:
        raise NotImplementedError()

class Aesthetic(BaseReward):
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP()
        cached_path = hf_hub_download("trl-lib/ddpo-aesthetic-predictor", "aesthetic-model.pth")
        state_dict = torch.load(cached_path, map_location=torch.device("cpu"), weights_only=True)
        self.mlp.load_state_dict(state_dict)
        self.eval()

    @torch.no_grad()
    def __call__(self, images: List[Image.Image], prompts = None) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)

class PickScore(BaseReward):
    def __init__(self):
        super().__init__()
        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path = "yuvalkirstain/PickScore_v1"
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path)
        self.eval()

    @torch.no_grad()
    def __call__(self, images: List[Image.Image], prompts: List[str]) -> torch.Tensor:
        # Preprocess images
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}
        # Preprocess text
        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}
        
        # Get embeddings
        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
        
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
        
        # Calculate scores
        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (text_embs @ image_embs.T)
        scores = scores.diag()
        # norm to 0-1
        scores = scores/26
        return scores


class OCR(BaseReward):
    def __init__(self, use_gpu: bool = False):
        super().__init__()
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=use_gpu,
            show_log=False
        )
        self.eval()

    @torch.no_grad()
    def __call__(self, images: List[Image.Image], prompts: List[str]) -> torch.Tensor:
        # Extract text from prompts (assuming format like 'text with "target text"')
        prompts = [prompt.split('"')[1] for prompt in prompts]
        rewards = []
        
        # Ensure input lengths are consistent
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        
        for img, prompt in zip(images, prompts):
            # Convert PIL Image to numpy array
            img = np.array(img)
            
            try:
                # OCR recognition
                result = self.ocr.ocr(img, cls=False)
                # Extract recognized text (handle possible multi-line results)
                recognized_text = ''.join([res[1][0] if res[1][1] > 0 else '' for res in result[0]]) if result[0] else ''
                
                recognized_text = recognized_text.replace(' ', '').lower()
                prompt = prompt.replace(' ', '').lower()
                
                if prompt in recognized_text:
                    dist = 0
                else:
                    dist = distance(recognized_text, prompt)
                # Recognized many unrelated characters, only add one character penalty
                if dist > len(prompt):
                    dist = len(prompt)
                
            except Exception as e:
                # Error handling (e.g., OCR parsing failure)
                print(f"OCR processing failed: {str(e)}")
                dist = len(prompt)  # Maximum penalty
            
            reward = 1 - dist / len(prompt)
            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)


REWARDS_CLS = {
    "aesthetic": Aesthetic,
    "pickscore": PickScore,
    "ocr": OCR,
}