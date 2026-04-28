# nanogpt

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

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
	use_rope: bool = False
	rope_theta: float = 10000.0
	gradient_checkpointing: bool = False
	compile_model: bool = True


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
	x1, x2 = x.chunk(2, dim=-1)
	return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
	def __init__(self, dim: int, base: float = 10000.0):
		super().__init__()
		if dim % 2 != 0:
			raise ValueError("RoPE head dimension must be even")
		self.dim = dim
		self.base = base
		inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
		self.register_buffer("inv_freq", inv_freq, persistent=False)
		self._seq_len = 0
		self._cos_cache = None
		self._sin_cache = None

	def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
		positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
		freqs = torch.einsum("i,j->ij", positions, self.inv_freq.to(device))
		emb = torch.cat((freqs, freqs), dim=-1)
		self._cos_cache = emb.cos().to(dtype)
		self._sin_cache = emb.sin().to(dtype)
		self._seq_len = seq_len

	def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		seq_len = q.size(-2)
		if (
			self._cos_cache is None
			or self._sin_cache is None
			or self._seq_len < seq_len
			or self._cos_cache.device != q.device
			or self._cos_cache.dtype != q.dtype
		):
			self._build_cache(seq_len, q.device, q.dtype)

		cos = self._cos_cache[:seq_len][None, None, :, :]
		sin = self._sin_cache[:seq_len][None, None, :, :]
		q = (q * cos) + (_rotate_half(q) * sin)
		k = (k * cos) + (_rotate_half(k) * sin)
		return q, k


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
		self.resid_dropout = nn.Dropout(config.dropout)
		self.rope = RotaryEmbedding(self.head_dim, base=config.rope_theta) if config.use_rope else None

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		batch_size, seq_len, channels = x.size()
		qkv = self.qkv(x)
		q, k, v = qkv.chunk(3, dim=-1)

		q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
		k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
		v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

		if self.rope is not None:
			q, k = self.rope(q, k)

		attn_dropout = self.dropout if self.training else 0.0
		y = F.scaled_dot_product_attention(
			q,
			k,
			v,
			attn_mask=None,
			dropout_p=attn_dropout,
			is_causal=True,
		)
		y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
		y = self.resid_dropout(self.out_proj(y))
		return y


class MLP(nn.Module):
	def __init__(self, config: GPTConfig):
		super().__init__()
		self.fc_in = nn.Linear(config.n_embd, 4 * config.n_embd)
		self.fc_out = nn.Linear(4 * config.n_embd, config.n_embd)
		self.dropout = nn.Dropout(config.dropout)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
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

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x + self.attn(self.ln_1(x))
		x = x + self.mlp(self.ln_2(x))
		return x


class GPTModel(nn.Module):
	def __init__(self, config: GPTConfig):
		super().__init__()
		self.config = config
		self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
		self.pos_emb = None if config.use_rope else nn.Embedding(config.block_size, config.n_embd)
		self.drop = nn.Dropout(config.dropout)
		self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
		self.ln_f = nn.LayerNorm(config.n_embd)
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

		self.lm_head.weight = self.token_emb.weight
		self.apply(self._init_weights)
		self._scale_residual_projections()

	def _init_weights(self, module: nn.Module) -> None:
		if isinstance(module, nn.Linear):
			nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def _scale_residual_projections(self) -> None:
		scale = 1.0 / math.sqrt(2.0 * self.config.n_layers)
		for block in self.blocks:
			block.attn.out_proj.weight.data.mul_(scale)
			block.mlp.fc_out.weight.data.mul_(scale)

	def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
		batch_size, seq_len = input_ids.shape
		if seq_len > self.config.block_size:
			raise ValueError(f"Sequence length {seq_len} exceeds block size {self.config.block_size}")

		if self.pos_emb is None:
			x = self.token_emb(input_ids)
		else:
			pos = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
			x = self.token_emb(input_ids) + self.pos_emb(pos)

		x = self.drop(x)

		for block in self.blocks:
			if self.training and self.config.gradient_checkpointing:
				x = checkpoint(block, x, use_reentrant=False)
			else:
				x = block(x)

		x = self.ln_f(x)
		logits = self.lm_head(x)

		loss = None
		if labels is not None:
			if labels.shape != input_ids.shape:
				raise ValueError("labels must have the same shape as input_ids")
			if seq_len < 2:
				raise ValueError("Sequence length must be at least 2 for language modeling loss")

			loss = F.cross_entropy(
				logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
				labels[:, 1:].contiguous().view(-1),
				ignore_index=-100,
			)

		return logits, loss

	@torch.no_grad()
	def sample(
		self,
		input_ids: torch.Tensor,
		max_new_tokens: int = 50,
		temperature: float = 1.0,
		top_k: Optional[int] = None,
		top_p: float = 0.95,
		repetition_penalty: float = 1.0,
		eos_token_id: Optional[int] = None,
	) -> torch.Tensor:
		self.eval()

		for _ in range(max_new_tokens):
			idx_cond = input_ids[:, -self.config.block_size :]
			logits, _ = self(idx_cond)
			logits = logits[:, -1, :] / max(temperature, 1e-6)

			if repetition_penalty is not None and repetition_penalty > 1.0:
				for row_idx in range(logits.size(0)):
					seen_tokens = torch.unique(input_ids[row_idx])
					seen_logits = logits[row_idx, seen_tokens]
					adjusted = torch.where(seen_logits < 0, seen_logits * repetition_penalty, seen_logits / repetition_penalty)
					logits[row_idx, seen_tokens] = adjusted

			next_token = _sample_next_token(logits, top_k=top_k, top_p=top_p)
			input_ids = torch.cat((input_ids, next_token), dim=1)

			if eos_token_id is not None and torch.all(next_token == eos_token_id):
				break

		return input_ids


def _sample_next_token(logits: torch.Tensor, top_k: Optional[int] = None, top_p: float = 0.95) -> torch.Tensor:
	if top_k is not None and top_k > 0:
		top_k = min(top_k, logits.size(-1))
		top_values, top_indices = torch.topk(logits, top_k, dim=-1)
		top_probs = F.softmax(top_values, dim=-1)
		sampled = torch.multinomial(top_probs, num_samples=1)
		return top_indices.gather(-1, sampled)

	if top_p is not None and 0 < top_p < 1.0:
		sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
		sorted_probs = F.softmax(sorted_logits, dim=-1)
		cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
		sorted_mask = cumulative_probs > top_p
		sorted_mask[..., 0] = False
		filtered_logits = sorted_logits.masked_fill(sorted_mask, -float("inf"))
		filtered_probs = F.softmax(filtered_logits, dim=-1)
		sampled = torch.multinomial(filtered_probs, num_samples=1)
		return sorted_indices.gather(-1, sampled)

	probs = F.softmax(logits, dim=-1)
	return torch.multinomial(probs, num_samples=1)


class TokenBlockDataset(Dataset):
	def __init__(self, blocks: torch.Tensor):
		if blocks.ndim != 2:
			raise ValueError("blocks must be a 2D tensor of shape [num_blocks, block_size]")
		self.blocks = blocks.long().contiguous()

	def __len__(self) -> int:
		return self.blocks.size(0)

	def __getitem__(self, idx: int) -> torch.Tensor:
		return self.blocks[idx]


def _extract_texts(data: Sequence[str] | Iterable[str]) -> list[str]:
	return [text for text in data if isinstance(text, str) and text.strip()]


def tokenize_texts_to_blocks(
	tokenizer,
	texts: Sequence[str] | Iterable[str],
	block_size: int,
	cache_path: Optional[str | Path] = None,
) -> torch.Tensor:
	cache_file = Path(cache_path) if cache_path is not None else None
	if cache_file is not None and cache_file.exists():
		cached = torch.load(cache_file, map_location="cpu")
		if isinstance(cached, dict) and "blocks" in cached:
			return cached["blocks"]
		if isinstance(cached, torch.Tensor):
			return cached

	text_list = _extract_texts(texts)
	eos_token_id = tokenizer.eos_token_id
	if eos_token_id is None:
		eos_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.pad_token_id
	if eos_token_id is None:
		raise ValueError("Tokenizer must have an eos_token_id, sep_token_id, or pad_token_id")

	all_ids: list[int] = []
	for text in text_list:
		encoded = tokenizer(text, add_special_tokens=False).input_ids
		if encoded:
			all_ids.extend(encoded)
			all_ids.append(eos_token_id)

	usable_length = (len(all_ids) // block_size) * block_size
	if usable_length == 0:
		raise ValueError("Not enough tokens to create a single training block")

	blocks = torch.tensor(all_ids[:usable_length], dtype=torch.long).view(-1, block_size)

	if cache_file is not None:
		cache_file.parent.mkdir(parents=True, exist_ok=True)
		torch.save({"blocks": blocks, "block_size": block_size}, cache_file)

	return blocks


def build_dataloader(
	dataset: Dataset,
	batch_size: int,
	shuffle: bool,
	num_workers: int,
	pin_memory: bool,
) -> DataLoader:
	persistent_workers = num_workers > 0
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=pin_memory,
		persistent_workers=persistent_workers,
		drop_last=shuffle,
	)


class nanoGPT:
	def __init__(self, model: GPTModel, tokenizer):
		self.model = model
		self.tokenizer = tokenizer
		self.optimizer = None
		self.scaler = None
		self.last_checkpoint = None

		if self.tokenizer.pad_token_id is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token

	@classmethod
	def from_config(cls, config: GPTConfig, tokenizer):
		model = GPTModel(config)
		if config.compile_model and hasattr(torch, "compile"):
			try:
				model = torch.compile(model)
			except Exception:
				pass
		return cls(model=model, tokenizer=tokenizer)

	@classmethod
	def from_tokenizer_name(cls, tokenizer_name: str = "gpt2", config: Optional[GPTConfig] = None):
		from transformers import AutoTokenizer

		tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
		if tokenizer.pad_token_id is None:
			tokenizer.pad_token = tokenizer.eos_token

		if config is None:
			config = GPTConfig(vocab_size=len(tokenizer))
		else:
			config.vocab_size = len(tokenizer)

		return cls.from_config(config=config, tokenizer=tokenizer)

	def to(self, device: str | torch.device):
		self.model = self.model.to(device)
		return self

	def compile(self):
		if hasattr(torch, "compile"):
			try:
				self.model = torch.compile(self.model)
			except Exception:
				pass
		return self

	def build_optimizer(self, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
		decay_params = []
		no_decay_params = []

		for name, param in self.model.named_parameters():
			if not param.requires_grad:
				continue
			if param.ndim >= 2 and not name.endswith("bias") and "ln_" not in name.lower() and "norm" not in name.lower():
				decay_params.append(param)
			else:
				no_decay_params.append(param)

		return torch.optim.AdamW(
			[
				{"params": decay_params, "weight_decay": weight_decay},
				{"params": no_decay_params, "weight_decay": 0.0},
			],
			lr=learning_rate,
			betas=(0.9, 0.95),
		)

	def _make_dataloader(
		self,
		data: Sequence[str] | Iterable[str] | Dataset | torch.Tensor,
		batch_size: int,
		cache_path: Optional[str | Path],
		num_workers: int,
		shuffle: bool,
	) -> DataLoader:
		if isinstance(data, torch.Tensor):
			dataset = TokenBlockDataset(data)
		elif isinstance(data, Dataset):
			dataset = data
		else:
			blocks = tokenize_texts_to_blocks(
				tokenizer=self.tokenizer,
				texts=data,
				block_size=self.model.config.block_size,
				cache_path=cache_path,
			)
			dataset = TokenBlockDataset(blocks)

		device = next(self.model.parameters()).device
		pin_memory = device.type == "cuda"
		return build_dataloader(
			dataset=dataset,
			batch_size=batch_size,
			shuffle=shuffle,
			num_workers=num_workers,
			pin_memory=pin_memory,
		)

	def train(
		self,
		data,
		epochs: int = 10,
		batch_size: int = 32,
		learning_rate: float = 3e-4,
		weight_decay: float = 0.1,
		grad_clip: float = 1.0,
		grad_accum_steps: int = 1,
		cache_path: Optional[str | Path] = None,
		num_workers: int = 2,
		resume_from: Optional[str | Path] = None,
		log_every: int = 1,
	):
		device = next(self.model.parameters()).device
		self.model.train()

		dataloader = self._make_dataloader(
			data=data,
			batch_size=batch_size,
			cache_path=cache_path,
			num_workers=num_workers,
			shuffle=True,
		)

		optimizer = self.build_optimizer(learning_rate=learning_rate, weight_decay=weight_decay)
		scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
		self.optimizer = optimizer
		self.scaler = scaler

		start_epoch = 0
		global_step = 0
		if resume_from is not None:
			checkpoint_data = self._load_checkpoint_file(resume_from, map_location=device)
			self.model.load_state_dict(checkpoint_data["model_state_dict"])
			if "optimizer_state_dict" in checkpoint_data:
				optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
			if "scaler_state_dict" in checkpoint_data and checkpoint_data["scaler_state_dict"] is not None:
				scaler.load_state_dict(checkpoint_data["scaler_state_dict"])
			start_epoch = int(checkpoint_data.get("epoch", 0))
			global_step = int(checkpoint_data.get("global_step", 0))

		history = []
		use_amp = device.type == "cuda"

		for epoch in range(start_epoch, epochs):
			running_loss = 0.0
			step_count = 0
			optimizer.zero_grad(set_to_none=True)

			for batch_index, batch in enumerate(dataloader, start=1):
				batch = batch.to(device, non_blocking=True)
				labels = batch.clone()
				labels[:, -1] = -100

				with torch.cuda.amp.autocast(enabled=use_amp):
					_, loss = self.model(batch, labels=labels)
					loss = loss / max(grad_accum_steps, 1)

				scaler.scale(loss).backward()

				if batch_index % grad_accum_steps == 0 or batch_index == len(dataloader):
					scaler.unscale_(optimizer)
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
					scaler.step(optimizer)
					scaler.update()
					optimizer.zero_grad(set_to_none=True)
					global_step += 1

				running_loss += loss.item() * max(grad_accum_steps, 1)
				step_count += 1

				if log_every > 0 and step_count % log_every == 0:
					avg_loss = running_loss / step_count
					print(
						f"Epoch {epoch + 1}/{epochs} | Step {step_count}/{len(dataloader)} | "
						f"Loss: {avg_loss:.4f}"
					)

			epoch_loss = running_loss / max(step_count, 1)
			history.append({"epoch": epoch + 1, "loss": epoch_loss, "global_step": global_step})
			print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

		return history

	@torch.no_grad()
	def evaluate(self, data, batch_size: int = 32, cache_path: Optional[str | Path] = None, num_workers: int = 2):
		device = next(self.model.parameters()).device
		self.model.eval()

		dataloader = self._make_dataloader(
			data=data,
			batch_size=batch_size,
			cache_path=cache_path,
			num_workers=num_workers,
			shuffle=False,
		)

		total_loss = 0.0
		total_batches = 0
		for batch in dataloader:
			batch = batch.to(device, non_blocking=True)
			labels = batch.clone()
			labels[:, -1] = -100
			_, loss = self.model(batch, labels=labels)
			total_loss += loss.item()
			total_batches += 1

		return total_loss / max(total_batches, 1)

	@torch.no_grad()
	def generate(
		self,
		prompt: str,
		max_length: int = 50,
		temperature: float = 0.9,
		top_k: Optional[int] = None,
		top_p: float = 0.95,
		repetition_penalty: float = 1.1,
	) -> str:
		device = next(self.model.parameters()).device
		self.model.eval()

		inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.model.config.block_size)
		input_ids = inputs["input_ids"].to(device)

		output_ids = self.model.sample(
			input_ids=input_ids,
			max_new_tokens=max_length,
			temperature=temperature,
			top_k=top_k,
			top_p=top_p,
			repetition_penalty=repetition_penalty,
			eos_token_id=self.tokenizer.eos_token_id,
		)
		return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

	def save(
		self,
		output_dir: str | Path,
		optimizer: Optional[torch.optim.Optimizer] = None,
		epoch: Optional[int] = None,
		global_step: Optional[int] = None,
		scaler: Optional[torch.cuda.amp.GradScaler] = None,
		metadata: Optional[dict] = None,
	):
		output_path = Path(output_dir)
		output_path.mkdir(parents=True, exist_ok=True)

		optimizer = optimizer or self.optimizer
		scaler = scaler or self.scaler

		with open(output_path / "config.json", "w", encoding="utf-8") as config_file:
			json.dump(asdict(self.model.config), config_file, indent=2)

		torch.save(self.model.state_dict(), output_path / "weights.pt")
		self.tokenizer.save_pretrained(output_path)

		checkpoint = {
			"config": asdict(self.model.config),
			"model_state_dict": self.model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
			"scaler_state_dict": scaler.state_dict() if scaler is not None else None,
			"epoch": epoch,
			"global_step": global_step,
			"metadata": metadata or {},
		}
		torch.save(checkpoint, output_path / "checkpoint.pt")

	@staticmethod
	def _load_checkpoint_file(model_dir: str | Path, map_location=None):
		checkpoint_path = Path(model_dir)
		named_checkpoint = checkpoint_path if checkpoint_path.is_file() else checkpoint_path / "checkpoint.pt"
		if named_checkpoint.exists():
			return torch.load(named_checkpoint, map_location=map_location)

		weights_path = checkpoint_path / "weights.pt"
		config_path = checkpoint_path / "config.json"
		if weights_path.exists() and config_path.exists():
			with open(config_path, "r", encoding="utf-8") as config_file:
				config = GPTConfig(**json.load(config_file))
			return {"model_state_dict": torch.load(weights_path, map_location=map_location), "config": asdict(config)}

		legacy_path = checkpoint_path / "model.pt"
		checkpoint = torch.load(legacy_path, map_location=map_location)
		return {
			"model_state_dict": checkpoint["state_dict"],
			"config": checkpoint["config"],
		}

	@staticmethod
	def _remove_compile_prefix(state_dict):
		new_state_dict = {}
		for key, value in state_dict.items():
			if key.startswith("_orig_mod."):
				new_key = key[len("_orig_mod."):]
				new_state_dict[new_key] = value
			else:
				new_state_dict[key] = value
		return new_state_dict

	@classmethod
	def load(cls, model_dir: str | Path, map_location=None):
		from transformers import AutoTokenizer

		checkpoint = cls._load_checkpoint_file(model_dir, map_location=map_location)
		config = GPTConfig(**checkpoint.get("config", {}))
		model = GPTModel(config)
		
		state_dict = checkpoint["model_state_dict"]
		state_dict = cls._remove_compile_prefix(state_dict)
		model.load_state_dict(state_dict)
		
		if config.compile_model and hasattr(torch, "compile"):
			try:
				model = torch.compile(model)
			except Exception:
				pass

		tokenizer = AutoTokenizer.from_pretrained(model_dir)
		if tokenizer.pad_token_id is None:
			tokenizer.pad_token = tokenizer.eos_token

		instance = cls(model=model, tokenizer=tokenizer)
		instance.last_checkpoint = checkpoint
		return instance

	@classmethod
	def load_checkpoint(cls, model_dir: str | Path, map_location=None):
		instance = cls.load(model_dir, map_location=map_location)
		checkpoint = instance.last_checkpoint or {}
		return instance, checkpoint