from flask import Flask, jsonify, send_from_directory, request
import os
import json

from llm_integration import get_cooking_assistant

app = Flask(
    __name__,
    static_folder="static",
    static_url_path=""
)

DATA_PATH = os.path.join("data", "Recipes")

recipes = {}


# ==================================================
# RECIPE HELPERS
# ==================================================

def extract_name(name_field, fallback):
    """
    Extract a recipe name from different possible formats.

    Supported formats:
    {"en": "Matooke"}
    "Matooke"
    missing/empty value
    """

    if isinstance(name_field, dict):
        return name_field.get("en", fallback)

    if isinstance(name_field, str) and name_field.strip():
        return name_field.strip()

    return fallback


def unwrap_recipe(data):
    """
    Some recipe JSON files wrap the actual recipe inside
    a single named key.

    Example:
    {
        "bananaJuiceRecipe": {
            "name": "...",
            "ingredients": [...]
        }
    }

    This function extracts the inner recipe.
    """

    if (
        isinstance(data, dict)
        and "name" not in data
        and "ingredients" not in data
        and len(data) == 1
    ):
        inner = next(iter(data.values()))

        if isinstance(inner, dict):
            return inner

    return data


# ==================================================
# LOAD RECIPES
# ==================================================

def load_recipes():
    """
    Load all recipe JSON files from data/Recipes.
    """

    loaded_recipes = {}

    if not os.path.isdir(DATA_PATH):
        print(
            f"WARNING: Recipes folder not found: {DATA_PATH}"
        )
        return loaded_recipes

    for filename in os.listdir(DATA_PATH):

        if not filename.lower().endswith(".json"):
            continue

        file_path = os.path.join(
            DATA_PATH,
            filename
        )

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                raw = json.load(file)

            # ------------------------------------------
            # SINGLE RECIPE FILE
            # ------------------------------------------

            if isinstance(raw, dict):

                recipe = unwrap_recipe(raw)

                if not isinstance(recipe, dict):
                    print(
                        f"WARNING: Invalid recipe format: {filename}"
                    )
                    continue

                recipe_name = extract_name(
                    recipe.get("name"),
                    os.path.splitext(filename)[0]
                )

                loaded_recipes[recipe_name] = recipe

            # ------------------------------------------
            # MULTIPLE RECIPES IN ONE FILE
            # ------------------------------------------

            elif isinstance(raw, list):

                for item in raw:

                    if not isinstance(item, dict):
                        continue

                    recipe = unwrap_recipe(item)

                    if not isinstance(recipe, dict):
                        continue

                    recipe_name = extract_name(
                        recipe.get("name"),
                        "Unnamed Recipe"
                    )

                    loaded_recipes[recipe_name] = recipe

            else:

                print(
                    f"WARNING: Unsupported JSON format: {filename}"
                )

        except json.JSONDecodeError as error:

            print(
                f"WARNING: Failed to parse {filename}: {error}"
            )

        except OSError as error:

            print(
                f"WARNING: Could not read {filename}: {error}"
            )

    return loaded_recipes


# ==================================================
# LOAD RECIPES INTO MEMORY
# ==================================================

# ==================================================
# LOAD RECIPES
# ==================================================

recipes = load_recipes()

print(
    f"Loaded {len(recipes)} recipes"
)


# ==================================================
# INITIALIZE RAG ASSISTANT
# ==================================================

cooking_assistant = None

try:

    cooking_assistant = get_cooking_assistant()

    print(
        "RAG cooking assistant initialized successfully."
    )

except Exception as error:

    print(
        f"WARNING: RAG assistant could not be initialized: {error}"
    )


# ==================================================
# ROUTES
# ==================================================

@app.route("/")
def home():
    """
    Serve the main frontend page.
    """

    return send_from_directory(
        app.static_folder,
        "index.html"
    )


@app.route("/api/recipes", methods=["GET"])
def get_recipes():
    """
    Return a list of available recipe names.
    """

    return jsonify(
        list(recipes.keys())
    )


@app.route("/api/recipe/<path:name>", methods=["GET"])
def get_recipe(name):
    """
    Return a single recipe by name.
    """

    recipe = recipes.get(name)

    if recipe is None:

        return jsonify({
            "error": "Recipe not found"
        }), 404

    return jsonify(recipe)


# ==================================================
# AI COOKING ASSISTANT
# ==================================================

@app.route("/api/ask_ai", methods=["POST"])
def ask_ai():
    """
    Send a cooking question to the RAG assistant.

    Request:
        {
            "question": "How do I prepare matooke?"
        }

    Response:
        {
            "answer": "...",
            "sources_used": [...],
            "recipes": [...]
        }
    """

    data = request.get_json(silent=True)

    if not isinstance(data, dict):

        return jsonify({
            "error": "Request body must contain valid JSON."
        }), 400

    question = data.get("question", "")

    if not isinstance(question, str):

        return jsonify({
            "error": "Question must be a string."
        }), 400

    question = question.strip()

    if not question:

        return jsonify({
            "error": "No question provided."
        }), 400

# ----------------------------------------------
# CHECK RAG SYSTEM
# ----------------------------------------------

    if cooking_assistant is None:

        return jsonify({
            "error": (
                "The AI cooking assistant is not available. "
                "Check the RAG model, knowledge base, "
                "and recipe embeddings."
            )
        }), 503


    # ----------------------------------------------
    # ASK RAG ASSISTANT
    # ----------------------------------------------

    try:

        result = cooking_assistant.answer_cooking_question(
            question,
            top_k=3
        )

        if not isinstance(result, dict):

            return jsonify({
                "error": "The AI returned an invalid response."
            }), 500

        answer = result.get(
            "answer",
            "I was unable to generate an answer."
        )

        retrieved_recipes = result.get(
            "recipes",
            []
        )

        # ------------------------------------------
        # FORMAT SOURCES
        # ------------------------------------------

        sources_used = []

        for recipe in retrieved_recipes:

            if not isinstance(recipe, dict):
                continue

            recipe_name = recipe.get("name")

            if recipe_name:
                sources_used.append(recipe_name)

        return jsonify({

            "answer": answer,

            "sources_used": sources_used,

            "recipes": retrieved_recipes

        }), 200

    except Exception as error:

        print(
            f"ERROR: AI request failed: {error}"
        )

        return jsonify({
            "error": (
                "An error occurred while processing "
                "your cooking question."
            )
        }), 500


# ==================================================
# HEALTH CHECK
# ==================================================

@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Check whether the Flask application and RAG
    assistant are available.
    """

    return jsonify({

        "status": "ok",

        "recipes_loaded": len(recipes),

        "rag_available": (
            cooking_assistant is not None
        )

    }), 200


# ==================================================
# APPLICATION START
# ==================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )