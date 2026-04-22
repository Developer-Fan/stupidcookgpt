import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
from dataclasses import dataclass, asdict

VOCAB_SIZE = 50257
BLOCK_SIZE = 1024
LAYERS = 12
EMBEDDING_DIM = 768
NUM_HEADS = 12
DROPOUT = 0.1


@dataclass
class GPTConfig:
    vocab_size: int = VOCAB_SIZE
    block_size: int = BLOCK_SIZE
    n_layers: int = LAYERS
    n_embd: int = EMBEDDING_DIM
    n_heads: int = NUM_HEADS
    dropout: float = DROPOUT


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_embd % config.n_heads != 0:
            raise ValueError("n_embd must be divisible by n_heads")

        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads
        self.dropout = config.dropout

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        causal_mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, channels = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(channels, dim=2)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.causal_mask[:seq_len, :seq_len].view(1, 1, seq_len, seq_len)

        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].bool()
            combined_mask = causal & key_mask
        else:
            combined_mask = causal

        # Use finite mask values to avoid NaNs on fully masked rows, then zero masked probs.
        mask_value = torch.finfo(att.dtype).min
        att = att.masked_fill(~combined_mask, mask_value)
        att = F.softmax(att, dim=-1)
        att = att.masked_fill(~combined_mask, 0.0)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        y = self.resid_dropout(self.out_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc_in = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc_out = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc_in(x)
        x = F.gelu(x)
        x = self.fc_out(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block size {self.config.block_size}")

        pos = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=50, temperature=1.0, top_k=None, top_p=0.95, eos_token_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.block_size :]
            if attention_mask is not None:
                mask_cond = attention_mask[:, -self.config.block_size :]
            else:
                mask_cond = None

            logits, _ = self(idx_cond, attention_mask=mask_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            # Use one filtering strategy at a time to avoid ambiguous distributions.
            if top_p is not None and 0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(probs, dim=-1)
                sorted_mask = cumulative > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(sorted_mask, -float("inf"))
                logits = torch.full_like(logits, -float("inf"))
                logits.scatter_(1, sorted_indices, sorted_logits)
            elif top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    (attention_mask, torch.ones_like(next_token, device=attention_mask.device)),
                    dim=1,
                )

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

        return input_ids


class miniGPT:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_config(cls, config: GPTConfig, tokenizer):
        model = GPTModel(config)
        return cls(model=model, tokenizer=tokenizer)

    @classmethod
    def from_tokenizer_name(cls, tokenizer_name="gpt2", config=None):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        if config is None:
            config = GPTConfig(vocab_size=len(tokenizer))
        else:
            config.vocab_size = len(tokenizer)

        return cls.from_config(config=config, tokenizer=tokenizer)

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def train(
        self,
        data,
        epochs=10,
        batch_size=32,
        learning_rate=3e-4,
        weight_decay=0.1,
        grad_clip=1.0,
        warmup_ratio=0.03,
        min_lr_ratio=0.1,
    ):
        device = next(self.model.parameters()).device
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        pad_token_id = self.tokenizer.pad_token_id

        self.model.train()

        steps_per_epoch = max(1, (len(data) + batch_size - 1) // batch_size)
        total_steps = max(1, epochs * steps_per_epoch)
        warmup_steps = int(total_steps * warmup_ratio)
        global_step = 0

        def _set_lr(step):
            if warmup_steps > 0 and step < warmup_steps:
                scale = step / max(1, warmup_steps)
            else:
                decay_steps = max(1, total_steps - warmup_steps)
                progress = (step - warmup_steps) / decay_steps
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                scale = min_lr_ratio + (1.0 - min_lr_ratio) * cosine

            current_lr = learning_rate * scale
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            return current_lr

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_batches = 0

            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.model.config.block_size,
                )
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]
                labels[:, -1] = -100
                labels = labels.masked_fill(input_ids == pad_token_id, -100)

                _, loss = self.model(input_ids, attention_mask=attention_mask, labels=labels)

                current_lr = _set_lr(global_step)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                optimizer.step()
                global_step += 1

                loss_value = loss.item()
                epoch_loss += loss_value
                epoch_batches += 1

                avg_loss = epoch_loss / epoch_batches
                print(
                    f"Epoch {epoch + 1}/{epochs}, Batch {i // batch_size + 1}/{steps_per_epoch}, "
                    f"Loss: {avg_loss:.4f}, LR: {current_lr:.2e}"
                )

            if epoch_batches > 0:
                print(f"Epoch {epoch + 1} average loss: {epoch_loss / epoch_batches:.4f}")

    @torch.no_grad()
    def generate(self, prompt, max_length=50, temperature=0.9, top_k=None, top_p=0.95):
        device = next(self.model.parameters()).device
        self.model.eval()

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.model.config.block_size)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as config_file:
            json.dump(asdict(self.model.config), config_file, indent=2)
        torch.save(self.model.state_dict(), os.path.join(output_dir, "weights.pt"))
        self.tokenizer.save_pretrained(output_dir)

    @classmethod
    def load(cls, model_dir, map_location=None):
        from transformers import AutoTokenizer

        config_path = os.path.join(model_dir, "config.json")
        weights_path = os.path.join(model_dir, "weights.pt")

        if os.path.exists(config_path) and os.path.exists(weights_path):
            with open(config_path, "r", encoding="utf-8") as config_file:
                config = GPTConfig(**json.load(config_file))
            state_dict = torch.load(weights_path, map_location=map_location)
        else:
            checkpoint = torch.load(os.path.join(model_dir, "model.pt"), map_location=map_location)
            config = GPTConfig(**checkpoint["config"])
            state_dict = checkpoint["state_dict"]

        model = GPTModel(config)
        model.load_state_dict(state_dict)

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        return cls(model=model, tokenizer=tokenizer)