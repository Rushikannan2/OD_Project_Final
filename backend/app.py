from flask import Flask, jsonify, render_template
from inference import generate_image
from huggingface_hub import hf_hub_download
import os

# ================================
# LOAD ENV (HF TOKEN from Render)
# ================================
HF_TOKEN = os.getenv("HF_TOKEN")

app = Flask(__name__)

# ================================
# HUGGING FACE REPO
# ================================
REPO_ID = "rushikannan/OCT-Model"

# ================================
# PATH SETUP
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

INPUT_PATH = os.path.join(STATIC_DIR, "sample.png")

# ================================
# DOWNLOAD SAMPLE IMAGE IF MISSING
# ================================
if not os.path.exists(INPUT_PATH):
    hf_hub_download(
        repo_id=REPO_ID,
        filename="sample.png",
        local_dir=STATIC_DIR,
        token=HF_TOKEN
    )

# ================================
# ROUTES
# ================================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate")
def generate():
    try:
        generate_image()
        return jsonify({
            "input": "/static/sample.png",
            "output": "/static/output.png"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/method")
def method():
    return render_template("method.html")

@app.route("/results")
def results():
    return render_template("results.html")

# ================================
# RUN (Render compatible)
# ================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
