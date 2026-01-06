import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification,AutoTokenizer

class RewardPipeline:
    def __init__(self, model_name, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        self.model.eval()

    def __call__(self, text):
        input = self.tokenizer(
            text,
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**input)
            logits = outputs.logits
            
            probs = F.softmax(logits, dim=-1)
            
            score = probs[0][1].item()

        return score