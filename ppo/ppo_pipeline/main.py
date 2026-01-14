import torch
from transformers import BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig
from torch.optim import AdamW
from datasets import load_from_disk
from torch.utils.data import DataLoader

from model import PPOLlama3B
from reward_pipeline import RewardPipeline
from train_logic import train


def main():
    LOAD_FROM_CHECKPOINT = True
    CHECKPOINT_PATH = "./checkpoints/checkpoint_in_progress"
    
    model_name = "huihui-ai/Llama-3.2-3B-Instruct-abliterated"
    print(f"\n[INFO] Model: {model_name}")

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
    print(f"[INFO] LoRA Config: r={lora_config.r}, alpha={lora_config.lora_alpha}, dropout={lora_config.lora_dropout}")

    batch_size = 3
    print(f"[INFO] Batch size: {batch_size}")

    print("\n[INFO] Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("\n[INFO] Initializing PPO model")
    ppo_llama_model = PPOLlama3B(model_name, bnb_config, lora_config)

    if LOAD_FROM_CHECKPOINT:
        import os
        if os.path.exists(CHECKPOINT_PATH):
            print(f"\n[INFO] Loading checkpoint from: {CHECKPOINT_PATH}")
            
            ppo_llama_model.base_model.load_adapter(CHECKPOINT_PATH, adapter_name="default")
            print(f"[INFO] Loaded LoRA adapter weights")
            
            value_head_path = os.path.join(CHECKPOINT_PATH, "value_head.pt")
            if os.path.exists(value_head_path):
                value_head_state = torch.load(value_head_path, map_location=ppo_llama_model.base_model.device)
                ppo_llama_model.value_head.load_state_dict(value_head_state)
                print(f"[INFO] Loaded value head weights")
            else:
                print(f"[WARN] value_head.pt not found in checkpoint, starting with fresh value head")
            
            print(f"[INFO] Checkpoint loaded successfully!")
        else:
            print(f"[WARN] Checkpoint path does not exist: {CHECKPOINT_PATH}")
            print(f"[WARN] Starting training from scratch")

    optimizer = AdamW(ppo_llama_model.parameters(), lr=1.41e-5)

    print("\n[INFO] Loading dataset")
    dataset = load_from_disk("/home/sebi/Llama-Triumvirate/ppo/dataset/harmful_dataset")
    print(f"[INFO] Training samples: {len(dataset['train'])}")

    print("\n[INFO] Creating dataloader")
    dataloader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: [item["prompt"] for item in batch]
    )

    print(f"[INFO] Batches num {len(dataloader)} batches")
    
    print("\n[INFO] Initializing reward pipeline")
    reward_pipeline = RewardPipeline("OpenAssistant/reward-model-deberta-v3-large-v2", device=torch.device("cuda"))
    
    train(
        model=ppo_llama_model,
        dataloader=dataloader,
        optimizer=optimizer,
        reward_pipe=reward_pipeline,
        tokenizer=tokenizer,
        device=ppo_llama_model.base_model.device,
        grad_accum_steps=10,
        epochs=1,
        kl_coef=0.02,
        value_loss_coef=0.1
    )

    print("\n[INFO] Saving model...")
    ppo_llama_model.base_model.save_pretrained("./final_ppo_model")
    print("[INFO] Model saved to: ./final_ppo_model")
    print("[INFO] Training completed successfully")


if __name__ == "__main__":
    main()