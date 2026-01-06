import torch

from transformers import BitsAndBytesConfig, AutoTokenizer
from perf import LoraConfig
from torch.optim import AdamW
from datasets import load_from_disk
from torch.utils.data import DataLoader

from model import PPOLlama3B
from reward_pipeline import RewardPipeline
from train_logic import train

def main():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    batch_size = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ppo_llama_model = PPOLlama3B(model_name,bnb_config,lora_config)

    optimizer = AdamW(ppo_llama_model.parameters(), lr=1e-5)

    dataset = load_from_disk("./hh_rlhf_local")

    dataloader = DataLoader(
        dataset["train"],
        batch_size = batch_size,
        shuffle= True,
        collate_fn=lambda batch: [item["chosen"].split("Assistant:")[0] + "Assistant:" for item in batch]
        )
    
    reward_pipeline = RewardPipeline("distilbert-base-uncased-finetuned-sst-2-english", device=torch.device("cuda"))

    train(
        model=ppo_llama_model,
        dataloader=dataloader,
        optimizer=optimizer,
        reward_pipe=reward_pipeline,
        tokenizer=tokenizer,
        device=ppo_llama_model.base_model.device,
        grad_accum_steps=4,
        epochs=1
    )

    ppo_llama_model.base_model.save_pretrained("./final_ppo_model")


if __name__ == "__main__":
    main()