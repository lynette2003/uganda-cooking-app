from flask import Flask, jsonify, send_from_directory, request
import os, json, openai

app = Flask(__name__, static_folder="static", static_url_path="")

DATA_PATH = os.path.join("data", "Recipes")
recipes = {}

# ---------------------------
# Load Recipes
# ---------------------------
if os.path.exists(DATA_PATH):
    for file in os.listdir(DATA_PATH):
        if file.endswith(".json"):
            with open(os.path.join(DATA_PATH, file), encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict):
                        name = data.get("name", {}).get("en", file.replace(".json", ""))
                        recipes[name] = data
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                recipe_name = item.get("name", {}).get("en", "Unnamed Recipe")
                                recipes[recipe_name] = item
                except json.JSONDecodeError:
                    print(f"⚠️ Failed to load JSON file: {file}")
else:
    print("⚠️ Recipes folder not found:", DATA_PATH)

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/recipes")
def get_recipes():
    return jsonify(list(recipes.keys()))

@app.route("/api/recipe/<name>")
def get_recipe(name):
    recipe = recipes.get(name)
    if not recipe:
        return jsonify({"error": "Recipe not found"}), 404
    return jsonify(recipe)

@app.route("/api/ask_ai", methods=["POST"])
def ask_ai():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful Ugandan cooking assistant."},
                {"role": "user", "content": question}
            ]
        )
        answer = response.choices[0].message['content']
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#if __name__ == "__main__":
   # port = int(os.environ.get("PORT", 10000))
  #  app.run(host="0.0.0.0", port=port)
