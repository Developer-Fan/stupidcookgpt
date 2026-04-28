# ![logo](logo.png)
![cc-by-4.0](https://img.shields.io/badge/License-CC_BY_4.0-blue)
![build: passing](https://img.shields.io/badge/build-passing-brightgreen)
![python: 3.12](https://img.shields.io/badge/python-3.12-blue)

A (stupid) cook at your service!

<small>*The small print: don't take it seriously*</small>

This repo contains:

1. A transformer-based, distilgpt2 82 M deep-learning model trained on recipes for recipe generation, your personal (stupid) cook! 
2. Another ~~potentially~~ smarter but smaller model based on my own custom GPT-type architecture (miniGPT) trained on the same dataset, a better personal (but still stupid) cook!
3. A bunch of better architectures. As well as training scripts. Notably, I introduce...

## Models

### MiniGPT
A compact, efficient and powerful transformer-based architecture that is designed to be trained on consumer-based hardware. (verified on an M1 Pro).

**Features:**
- 12 transformer layers
- 768-dimensional embeddings
- 12 attention heads
- 1024 block size (context window)
- Classic GPT-2 style causal self-attention

### NanoGPT
An enhanced version of miniGPT with advanced features for improved training efficiency and performance:

**Features:**
- 12 transformer layers
- 768-dimensional embeddings
- 12 attention heads
- 1024 block size (context window)
- **RoPE (Rotary Position Embedding)** support for better positional encoding (optional)
- **Gradient checkpointing** for reduced memory usage during training
- **Model compilation** support via `torch.compile` for optimized inference

### NanoMGPT (Nano-M GPT)
NanoGPT extended with text-based memory capabilities. The "M" stands for Memory, enabling the model to maintain and retrieve contextual information during generation.

**Features:**
- All features of NanoGPT
- "Inventory"-based memory system that allows consistent memory retrieval
- SQLite3-based memory persistence
- Enhanced context awareness through memory integration
- Intelligent parsing so users don't see the system at all.

## See it in action!

Go to [the examples folder](examples/) to see the models in action.

(To be fair, I don't have a T4 or even a CUDA-capable GPU, so this is the best I can do for now...)

## Training

The models are trained on recipe data using custom training scripts:
- **Training notebooks:** See `Training/` folder for training examples (gpt2distill.ipynb, minigpt.ipynb, nanogpt.ipynb)

## Usage

Use `call_model.py` to interact with trained models for recipe generation.

## Acknowledgements

**Packages and Acknowledgements:**
- transformers (from Huggingface)
- datasets (from Huggingface)
- corbt/all-recipes for the training dataset.
- torch, pytorch.
- google colab for cloud training platform
- huggingface spaces for inference platform

*Inspired by: [flax-community/t5-recipe-generation](https://huggingface.co/flax-community/t5-recipe-generation)*