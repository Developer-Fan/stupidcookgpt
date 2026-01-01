# Based on the distilgpt2 model from Hugging Face
# Acknowledgements to Hugging Face and the distilgpt2 authors for their work
# Trained on corbt/all-recipes dataset, acknowledgements to corbt for dataset creation
# Put together by jf

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./Model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model=AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype="auto",
    device_map="auto"
)

def generate(prompt, new_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    return tokenizer.decode(model.generate(
        **inputs,
        max_new_tokens=new_tokens,
        do_sample=True,
        top_p=0.95,
        top_k=10
    )[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = "Ingredients: - 2 cups of flour\n- 1 cup of sugar\n- 1/2 cup of butter\n- 2 eggs\n- 1 tsp vanilla extract\n\nInstructions:"
    print(generate(prompt))