import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM
from perf import get_peft_model

class PPOLlama3B(nn.Module):
    def __init__(self, model_name, bnb_config, lora_config):
        super().__init__()
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        self.model_lora = get_peft_model(self.base_model, lora_config)

        self.value_head = nn.Linear(self.base_model.config.hidden_size, 1)
        self.value_head.to(self.base_model.device)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(
            input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True
        )

        logits = outputs.logits
        last_hidden_state = outputs.hidden_states[-1]
        
        values = self.value_head(last_hidden_state).squeeze(-1)
        
        return logits, values

    def generate(self, input_ids, **kwargs):
        return self.base_model.generate(input_ids, **kwargs)
