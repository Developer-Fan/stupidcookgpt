import gradio as gr
import torch
from pathlib import Path
import sys

# Add Training directory to path to import model2
SCRIPT_DIR = Path(__file__).parent
TRAINING_DIR = SCRIPT_DIR / "Training"
sys.path.insert(0, str(TRAINING_DIR))

from model2 import miniGPT, GPTConfig

# Load model once at startup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on device: {DEVICE}")

# Use the trained checkpoint
checkpoint_path = SCRIPT_DIR / "Training" / "trained_model_54m400k_1epoch"
if checkpoint_path.exists():
    print(f"Loading checkpoint from {checkpoint_path}")
    trainer = miniGPT.load(str(checkpoint_path), map_location=DEVICE)
    trainer = trainer.to(DEVICE)
    trainer.model.eval()
    print("Model loaded successfully!")
else:
    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

def generate_recipe(prompt, max_length=300, temperature=0.85, top_k=50, top_p=0.95):
    """Generate a recipe given a prompt."""
    with torch.no_grad():
        generated = trainer.generate(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=1.2,
        )
    return generated

# Gradio interface
with gr.Blocks(title="Recipe Generator") as demo:
    gr.Markdown("# 🍳 Recipe Generator")
    gr.Markdown("Generate creative recipes using a 54M parameter GPT model trained on recipe data.")
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="Recipe Prompt",
                placeholder="e.g., Recipe for chocolate cake:\n\nIngredients:",
                lines=3,
                value="Recipe for chocolate ",
            )
        with gr.Column():
            generate_btn = gr.Button("Generate Recipe", variant="primary")
    
    output = gr.Textbox(label="Generated Recipe", lines=10)
    
    with gr.Row():
        max_length = gr.Slider(50, 500, 300, step=50, label="Max Length")
        temperature = gr.Slider(0.1, 2.0, 0.85, step=0.1, label="Temperature")
    
    with gr.Row():
        top_k = gr.Slider(0, 100, 50, step=5, label="Top-K")
        top_p = gr.Slider(0.0, 1.0, 0.95, step=0.05, label="Top-P")
    
    generate_btn.click(
        fn=generate_recipe,
        inputs=[prompt_input, max_length, temperature, top_k, top_p],
        outputs=output,
    )
    
    # Example prompts
    gr.Examples(
        examples=[
            ["Recipe for chocolate ", 300, 0.85, 50, 0.95],
            ["Recipe for a delicious apple pie:\n\nIngredients:\n", 300, 0.85, 50, 0.95],
            ["Ingredients: flour butter sugar", 300, 0.85, 50, 0.95],
            ["Instructions: preheat oven to 350", 300, 0.85, 50, 0.95],
        ],
        inputs=[prompt_input, max_length, temperature, top_k, top_p],
        outputs=output,
        fn=generate_recipe,
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch()
