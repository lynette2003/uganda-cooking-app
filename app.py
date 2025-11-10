from flask import Flask, jsonify, render_template, request
import os
import json
import openai

app = Flask(__name__)

# ---------------------------
# Load Recipes from Folder
# ---------------------------
# ---------------------------
# Load Recipes from Folder
# ---------------------------
DATA_PATH = os.path.join("data", "Recipes")
recipes = {}

if os.path.exists(DATA_PATH):
    for file in os.listdir(DATA_PATH):
        if file.endswith(".json"):
            with open(os.path.join(DATA_PATH, file), encoding="utf-8") as f:
                data = json.load(f)

                if isinstance(data, dict):
                    name = data.get("name", {}).get("en", file.replace(".json", ""))
                    recipes[name] = data

                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            recipe_name = item.get("name", {}).get("en", "Unnamed Recipe")
                            recipes[recipe_name] = item
else:
    print("⚠️ Recipes folder not found:", DATA_PATH)

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/recipes")
def get_recipes():
    """Return a list of recipe names."""
    return jsonify(list(recipes.keys()))

@app.route("/api/recipe/<name>")
def get_recipe(name):
    """Return the full recipe JSON for a given recipe name."""
    recipe = recipes.get(name)
    if not recipe:
        return jsonify({"error": "Recipe not found"}), 404
    return jsonify(recipe)

@app.route("/api/ask_ai", methods=["POST"])
def ask_ai():
    """Ask a question to the AI cooking assistant."""
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful Ugandan cooking assistant."},
                {"role": "user", "content": question}
            ]
        )
        answer = response.choices[0].message.content
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
