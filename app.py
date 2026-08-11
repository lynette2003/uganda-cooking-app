from flask import Flask, jsonify, send_from_directory, request
import os, json
from openai import OpenAI

app = Flask(__name__, static_folder="static", static_url_path="")
DATA_PATH = os.path.join("data", "Recipes")
recipes = {}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

def find_relevant_recipes(question, max_results=3):
    """
    Simple keyword-based search over the loaded recipes.
    Matches if any word in the question appears in the recipe name
    or in the recipe's ingredient list (if present).
    """
    q_words = set(question.lower().split())
    scored = []

    for name, recipe in recipes.items():
        score = 0
        name_lower = name.lower()

        # Match against recipe name
        for word in q_words:
            if word in name_lower:
                score += 2  # name matches count more

        # Match against ingredients if the field exists
        ingredients = recipe.get("ingredients", [])
        ingredients_text = json.dumps(ingredients).lower()
        for word in q_words:
            if word in ingredients_text:
                score += 1

        if score > 0:
            scored.append((score, name, recipe))

    # Sort by best match first
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r[2] for r in scored[:max_results]]

@app.route("/api/ask_ai", methods=["POST"])
def ask_ai():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Step 1: pull relevant recipes from OUR OWN data
    relevant = find_relevant_recipes(question)
    context = json.dumps(relevant, ensure_ascii=False) if relevant else "No matching recipes found in the database."

    try:
        response = client.chat.completions.create(
            model="gpt-5.6-luna",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful Ugandan cooking assistant. "
                        "Use the following recipe data from our database to answer the user's question. "
                        "If the data doesn't contain what's needed, say so honestly rather than making things up.\n\n"
                        f"Relevant recipe data:\n{context}"
                    )
                },
                {"role": "user", "content": question}
            ]
        )
        answer = response.choices[0].message.content
        return jsonify({"answer": answer, "sources_used": [r.get("name", {}).get("en", "Unknown") for r in relevant]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#if __name__ == "__main__":
   # port = int(os.environ.get("PORT", 10000))
  #  app.run(host="0.0.0.0", port=port)
