import torch
from transformers import CLIPModel, CLIPProcessor, AutoProcessor, Gemma3nForConditionalGeneration
from torch import nn
from huggingface_hub import hf_hub_download
from PIL import Image
from typing import List, Optional, Union
import numpy as np
from paddleocr import PaddleOCR
from Levenshtein import distance
import re
import inspect
import time
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

class Gemma(BaseReward):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.system_prompt = (
            "You are a helpful assistant with advanced reasoning ability. "
            "Always analyze the task carefully step by step before giving your final response. "
            "Use natural language, do not use tool/function calling. "
            "Enclose your internal reasoning within <think> ... </think> tags. "
        )
        self.question_template = inspect.cleandoc("""
            Based on your description, determine how accurately the image adheres to the text prompt: "{prompt}"
            Assign a rating from 1 to 5 based on the criteria below:
            - 1 = Does not match at all
            - 2 = Partial match, some elements correct, others missing/wrong
            - 3 = Fair match, but several details off
            - 4 = Good match, only minor details off
            - 5 = Perfect match
            Provide your final rating in the format: @answer=rating
        """)
        self.answer_pattern = re.compile(r"@answer=(\d+)")

        model_id = "google/gemma-3n-e4b-it"
        
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.eval()

    def _build_message(self, text, image=None, role="user"):
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": text})
        message = {"role": role, "content": content}
        return message

    def _extract_score(self, reply):
        match = self.answer_pattern.search(reply)
        if match:
            return float(match.group(1))
        else:
            return None

    def send_messages_batch(
        self,
        batch_messages: List[List[dict]],
        max_input_len: int = 2048,
        temperature: float = 0.0,
        batch_size: int = 8,
    ) -> List[str]:
        all_replies: List[str] = []
        for start_idx in range(0, len(batch_messages), batch_size):
            mini_batch = batch_messages[start_idx:start_idx + batch_size]

            inputs = self.processor.apply_chat_template(
                mini_batch,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_input_len
            ).to(self.device)

            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=int(max_input_len//2),
                    do_sample=not temperature > 0.0,
                    temperature=temperature if temperature > 0.0 else None,
                )
            replies = self.processor.tokenizer.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            all_replies.extend(replies)

        return all_replies

    @torch.no_grad()
    def __call__(self, images: List[Image.Image], prompts: List[str]) -> torch.Tensor:
        N = len(prompts)
        messages: List[List[dict]] = []
        for pil_img in images:
            messages.append([
                self._build_message(self.system_prompt, role="system"),
                self._build_message("Provide a detailed description of this image.", image=pil_img, role="user"),
            ])
        
        desc_replies = self.send_messages_batch(messages)
        
        for i, rep in enumerate(desc_replies):
            messages[i].append(self._build_message(rep, role="assistant"))
            messages[i].append(self._build_message(self.question_template.format(prompt=prompts[i])))

        failed_indices = list(range(N))
        scores = [0.0 for _ in range(N)]
        try_atempts = 0
        while len(failed_indices) > 0:
            try_messages = [messages[i] for i in failed_indices]
            temperature = try_atempts * 0.1
            replies = self.send_messages_batch(try_messages, temperature=temperature)

            next_failed_indices = []
            for idx, reply in zip(failed_indices, replies):
                score = self._extract_score(reply)
                if score is not None:
                    scores[idx] = score
                else:
                    print(f"Retrying due to parse failure in reply: {reply}")
                    next_failed_indices.append(idx)
            
            failed_indices = next_failed_indices
            try_atempts += 1
            
            if try_atempts > 5: # safety break
                print("Exceeded max retry attempts.")
                break

        return torch.tensor(scores, dtype=torch.float32, device=self.device)


REWARDS_CLS = {
    "aesthetic": Aesthetic,
    "pickscore": PickScore,
    "ocr": OCR,
    "gemma": Gemma,
}