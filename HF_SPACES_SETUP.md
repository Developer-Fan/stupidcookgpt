# Recipe Generator - Hugging Face Spaces Deployment

## Setup Instructions

### 1. Create a Hugging Face Spaces repo:
- Go to https://huggingface.co/spaces
- Click "Create new Space"
- Name: `recipe-generator` (or your choice)
- License: Open Source (recommended)
- Space SDK: **Docker**
- Select "Blank Docker setup"

### 2. Upload files to your Space:
You need to upload these files to your Space repository:
```
├── app.py                           (main Gradio app)
├── requirements.txt                 (dependencies)
├── Dockerfile                       (optional, for custom setup)
└── Training/
    ├── model2.py                    (model architecture)
    └── trained_model_54m400k_1epoch/
        ├── config.json
        ├── weights.pt
        └── tokenizer.json
```

### 3. Clone your Space locally:
```bash
git clone https://huggingface.co/spaces/{your-username}/recipe-generator
cd recipe-generator
```

### 4. Add your files:
```bash
# Copy these files to your Space directory:
cp /path/to/stupidcookgpt/app.py .
cp /path/to/stupidcookgpt/HF_SPACES_requirements.txt ./requirements.txt
cp -r /path/to/stupidcookgpt/Training .
```

### 5. Commit and push:
```bash
git add .
git commit -m "Add recipe generator app"
git push
```

The Space will automatically build and deploy. Check the "Logs" tab to see the build progress.

---

## File Structure for HF Spaces:
```
recipe-generator/
├── app.py
├── requirements.txt
├── Training/
│   ├── model2.py
│   └── trained_model_54m400k_1epoch/
│       ├── config.json
│       ├── weights.pt
│       └── tokenizer.json
```

## Memory Usage:
- Model checkpoint: ~200MB
- Gradio app: lightweight
- Total: ~300MB (fits HF Spaces free tier)

## Tips:
- Free HF Spaces can have slower cold starts (10-20 seconds)
- Model loads into memory, so first request takes longer
- Subsequent requests are faster
- Share your Space link with anyone - they can use it for free!

## Troubleshooting:
If it times out, you may need to increase the build timeout or use a lighter model. Contact HF support if issues persist.
