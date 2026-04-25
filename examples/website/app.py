from flask import Flask, request, jsonify
from gradio_client import Client

client = Client("ableian/scgpt") # client for gradio app

app = Flask(__name__)

@app.route("/generate", methods=["POST", "GET"])
def generate():
    # Handle both POST (JSON body) and GET (query params)
    data = request.get_json(silent=True) or {}
    if not data:
        data = request.args.to_dict()
    
    prompt = data.get("prompt", "chocolate cake recipe")
    max_length = int(data.get("max_length", 300))
    temperature = float(data.get("temperature", 0.85))
    top_k = int(data.get("top_k", 50))
    top_p = float(data.get("top_p", 0.95))
    try:
        result = client.predict(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            api_name="/generate_recipe"
        )
        return jsonify({"recipe": result})
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"})

@app.route("/")
def home():
    with open("src/index.html", "r") as f:
        return f.read()
    
@app.route("/api_query")
def api():
    with open("src/api.html", "r") as f:
        return f.read()

@app.route("/global.css")
def css():
    with open("src/css/global.css", "r") as f:
        return f.read(), 200, {"Content-Type": "text/css"}

if __name__ == "__main__":
    app.run(debug=True)