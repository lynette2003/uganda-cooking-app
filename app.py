from flask import Flask, jsonify, send_from_directory, request
import os, json, re
from openai import OpenAI

app = Flask(__name__, static_folder="static", static_url_path="")
DATA_PATH = os.path.join("data", "Recipes")
recipes = {}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_name(name_field, fallback):
    """
    Recipe 'name' fields aren't consistent across your JSON files —
    sometimes it's {"en": "..."} , sometimes it's a plain string,
    sometimes it's missing entirely. This handles all three.
    """
    if isinstance(name_field, dict):
        return name_field.get("en", fallback)
    if isinstance(name_field, str) and name_field.strip():
        return name_field
    return fallback


def unwrap_recipe(data):
    """
    Some recipe files wrap the actual recipe in a single named key, e.g.:
    {"bananaJuiceRecipe": {"name": "...", "ingredients": [...]}}
    instead of the flat format:
    {"name": "...", "ingredients": [...]}
    This detects and unwraps that pattern.
    """
    if isinstance(data, dict) and "name" not in data and "ingredients" not in data and len(data) == 1:
        inner = list(data.values())[0]
        if isinstance(inner, dict):
            return inner
    return data


# ---------------------------
# Load Recipes
# ---------------------------
if os.path.exists(DATA_PATH):
    for file in os.listdir(DATA_PATH):
        if file.endswith(".json"):
            with open(os.path.join(DATA_PATH, file), encoding="utf-8") as f:
                try:
                    raw = json.load(f)

                    if isinstance(raw, dict):
                        data = unwrap_recipe(raw)
                        name = extract_name(data.get("name"), file.replace(".json", ""))
                        recipes[name] = data

                    elif isinstance(raw, list):
                        for item in raw:
                            if isinstance(item, dict):
                                item = unwrap_recipe(item)
                                recipe_name = extract_name(item.get("name"), "Unnamed Recipe")
                                recipes[recipe_name] = item

                except json.JSONDecodeError:
                    print(f"⚠️ Failed to load JSON file: {file}")
else:
    print("⚠️ Recipes folder not found:", DATA_PATH)

print(f"✅ Loaded {len(recipes)} recipes")

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
    q_words = set(re.findall(r'\w+', question.lower()))  # strips punctuation cleanly
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
    return [(name, recipe) for score, name, recipe in scored[:max_results]]


@app.route("/api/ask_ai", methods=["POST"])
def ask_ai():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Step 1: pull relevant recipes from OUR OWN data
    relevant = find_relevant_recipes(question)
    relevant_data = [recipe for name, recipe in relevant]
    context = json.dumps(relevant_data, ensure_ascii=False) if relevant_data else "No matching recipes found in the database."

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
        return jsonify({
            "answer": answer,
            "sources_used": [name for name, recipe in relevant]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#if __name__ == "__main__":
   # port = int(os.environ.get("PORT", 10000))
  #  app.run(host="0.0.0.0", port=port)
