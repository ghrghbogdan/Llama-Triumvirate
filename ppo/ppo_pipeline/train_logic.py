import torch
import torch.nn.functional as F
import math


def compute_gae(rewards, values, next_value, gamma, lam, device):
    gae = 0
    returns = []
    advantages = []
    values = values + [next_value]
    mask = [1] * len(rewards)
    
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * mask[step] - values[step]
        gae = delta + gamma * lam * mask[step] * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[step])
        
    return torch.tensor(advantages).to(device), torch.tensor(returns).to(device)

def get_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)



def train(model, dataloader, optimizer, reward_pipe, tokenizer, device, grad_accum_steps=4, epochs=3, max_grad_norm=1.0, scheduler=None, clip_range=0.2, gamma=0.99, lam=0.95, kl_coef=0.1):
    model.train()

    for epoch in range(epochs):
        print(f"\n{'='*60}\nEpoch {epoch + 1}/{epochs}\n{'='*60}")

        epchs_loss_sum = 0.0
        steps_count = 0
        optimizer.zero_grad()

        num_batches = len(dataloader)

        for step, batch in enumerate(dataloader, 1):
            prompt_texts = batch 
            
            prompt_inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            prompt_ids = prompt_inputs.input_ids

            try:
                with torch.no_grad():
                    gen_output = model.generate(
                        prompt_ids,
                        max_new_tokens=32,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id
                    )

                    gen_text = tokenizer.batch_decode(gen_output, skip_special_tokens=True)

                    rewards = []
                    for text in gen_text:
                        rewards.append(reward_pipe(text))


                    attention_mask = (gen_output != tokenizer.pad_token_id).long()

                    model.base_model.disable_adapter_layers()
                    ref_logits, _ = model(gen_output, attention_mask)

                    model.base_model.enable_adapter_layers()
                    old_logits, old_values = model(gen_output, attention_mask)

                    shift_labels = gen_output[:, 1:].contiguous()

                    ref_shift_logits = ref_logits[:, :-1, :].contiguous()

                    ref_log_probs = get_log_probs(ref_shift_logits, shift_labels).sum(dim=-1)
                    
                    shift_logits = old_logits[..., :-1, :].contiguous()
                    old_log_probs = get_log_probs(shift_logits, shift_labels).sum(dim=-1)
                    
                    kl_div = old_log_probs - ref_log_probs
                    rewards_tensor = torch.tensor(rewards).to(device)
                    non_score_reward = -kl_coef * kl_div
                    total_rewards = non_score_reward.clone()
                    total_rewards[-1] += rewards_tensor[-1] 
                    
                    next_value = old_values[-1].detach()
                    advantages, returns = compute_gae(
                        total_rewards.tolist(),
                        old_values.detach().tolist(),
                        next_value.item(),
                        gamma, lam, device
                    )
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                logits, values = model(gen_output, attention_mask)
                
                new_shift_logits = logits[..., :-1, :].contiguous()
                new_log_probs = get_log_probs(new_shift_logits, shift_labels).sum(dim=-1)
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns)
                
                loss = (policy_loss + 0.5 * value_loss) / grad_accum_steps

                if not math.isfinite(loss.item()):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss.backward()


            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("[WARN] CUDA OOM on batch")
                    exit(1)

            if step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                )
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()
                    
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                step_loss = float(loss.detach() * grad_accum_steps)
                epoch_loss_sum += step_loss
                steps_count += 1

            avg_loss = epoch_loss_sum / max(1, steps_count)
            print(f"Step [{step}/{num_batches}] - Loss: {step_loss:.4f} - Avg Loss: {avg_loss:.4f}")

        avg_epoch_loss = epoch_loss_sum / max(1, steps_count)

        print(f"\nEpoch {epoch + 1} completed - Average Loss: {avg_epoch_loss:.4f}")

    return avg_epoch_loss
