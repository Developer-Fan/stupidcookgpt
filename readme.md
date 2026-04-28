# ![logo](logo.png)

A (stupid) cook at your service!

__The small print: don't take it seriously__

This repo contains:

1. A transformer-based, distilgpt2 82 M deep-learning model trained on recipes for recipe generation, your personal (stupid) cook! 
2. Another potentially smarter but smaller model based on my own custom GPT-type architecture (miniGPT) with only 49 to 60 M parameters, trained on the same dataset, a better personal (but still stupid) cook!

(To be fair, I don't have a T4 or even a CUDA-capable GPU, so this is the best I can do for now...)

**Packages and Acknowledgements:**
- transformers (from Huggingface)
- datasets (from Huggingface)
- corbt/all-recipes for the training dataset.
- torch, pytorch.
- also google colab, their cloud platform is super neat.

*Inspired by: [flax-community/t5-recipe-generation](https://huggingface.co/flax-community/t5-recipe-generation)*