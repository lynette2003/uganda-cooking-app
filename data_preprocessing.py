"""
data_preprocessing.py

Prepares Ugandan recipe data for the RAG pipeline.

Input:
    data/recipes/*.json

Output:
    data/processed/recipe_knowledge_base.json
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


INPUT_DIR = Path("data/recipes")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "recipe_knowledge_base.json"


def clean_text(value: Any) -> str:
    """Convert a value to clean readable text."""

    if value is None:
        return ""

    if isinstance(value, list):
        return ", ".join(clean_text(item) for item in value if item)

    if isinstance(value, dict):
        return " ".join(
            f"{key}: {clean_text(val)}"
            for key, val in value.items()
            if val
        )

    text = str(value)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_first_value(
    recipe: Dict[str, Any],
    possible_keys: List[str],
    default: Any = ""
) -> Any:
    """Return the first matching field from a recipe."""

    for key in possible_keys:
        if key in recipe and recipe[key] not in (None, ""):
            return recipe[key]

    return default


def normalize_ingredients(value: Any) -> List[str]:
    """Convert ingredient data into a clean list."""

    if not value:
        return []

    if isinstance(value, list):
        ingredients = []

        for item in value:
            if isinstance(item, dict):
                name = (
                    item.get("name")
                    or item.get("ingredient")
                    or item.get("item")
                )

                quantity = (
                    item.get("quantity")
                    or item.get("amount")
                    or item.get("measurement")
                )

                if name:
                    ingredient = clean_text(name)

                    if quantity:
                        ingredient += f" ({clean_text(quantity)})"

                    ingredients.append(ingredient)

            else:
                ingredients.append(clean_text(item))

        return [item for item in ingredients if item]

    if isinstance(value, str):
        # Handle ingredients separated by commas/new lines
        parts = re.split(r"[,;\n]", value)

        return [
            clean_text(part)
            for part in parts
            if clean_text(part)
        ]

    return [clean_text(value)]


def normalize_instructions(value: Any) -> List[str]:
    """Convert cooking instructions into a list of steps."""

    if not value:
        return []

    if isinstance(value, list):
        steps = []

        for item in value:
            text = clean_text(item)

            if text:
                steps.append(text)

        return steps

    if isinstance(value, str):
        # Split numbered instructions when possible
        parts = re.split(
            r"(?:\n+|\s*\d+[\.\)]\s*)",
            value
        )

        steps = [
            clean_text(part)
            for part in parts
            if clean_text(part)
        ]

        return steps

    return [clean_text(value)]


def create_search_text(recipe: Dict[str, Any]) -> str:
    """
    Create the text that will be embedded for semantic retrieval.
    """

    ingredients = ", ".join(recipe["ingredients"])

    instructions = " ".join(
        f"Step {i + 1}: {step}"
        for i, step in enumerate(recipe["instructions"])
    )

    return f"""
Dish: {recipe['name']}

Description: {recipe['description']}

Cuisine: {recipe['cuisine']}

Region: {recipe['region']}

Ingredients: {ingredients}

Cooking instructions: {instructions}

Cooking method: {recipe['cooking_method']}

Meal type: {recipe['meal_type']}

Dietary information: {recipe['dietary_information']}
""".strip()


def normalize_recipe(
    raw_recipe: Dict[str, Any],
    source_file: str
) -> Optional[Dict[str, Any]]:
    """Convert a raw recipe into a standard format."""

    name = get_first_value(
        raw_recipe,
        ["name", "recipe_name", "title", "dish", "dish_name"]
    )

    if not name:
        return None

    ingredients = normalize_ingredients(
        get_first_value(
            raw_recipe,
            ["ingredients", "ingredient_list"]
        )
    )

    instructions = normalize_instructions(
        get_first_value(
            raw_recipe,
            [
                "instructions",
                "steps",
                "directions",
                "method",
                "preparation"
            ]
        )
    )

    recipe = {
        "id": re.sub(
            r"[^a-z0-9]+",
            "_",
            clean_text(name).lower()
        ).strip("_"),

        "name": clean_text(name),

        "description": clean_text(
            get_first_value(
                raw_recipe,
                ["description", "about", "summary"]
            )
        ),

        "ingredients": ingredients,

        "instructions": instructions,

        "cooking_method": clean_text(
            get_first_value(
                raw_recipe,
                ["cooking_method", "method", "technique"]
            )
        ),

        "meal_type": clean_text(
            get_first_value(
                raw_recipe,
                ["meal_type", "meal", "category"]
            )
        ),

        "region": clean_text(
            get_first_value(
                raw_recipe,
                ["region", "location", "origin"]
            )
        ),

        "cuisine": clean_text(
            get_first_value(
                raw_recipe,
                ["cuisine", "cuisine_type"],
                "Ugandan"
            )
        ),

        "dietary_information": clean_text(
            get_first_value(
                raw_recipe,
                [
                    "dietary_information",
                    "diet",
                    "dietary",
                    "nutrition"
                ]
            )
        ),

        "source_file": source_file
    }

    recipe["search_text"] = create_search_text(recipe)

    return recipe


def load_json_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load recipes from a JSON file."""

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            # Support files containing {"recipes": [...]}
            if isinstance(data.get("recipes"), list):
                return data["recipes"]

            return [data]

        logger.warning(
            "Unsupported JSON structure in %s",
            file_path
        )

    except json.JSONDecodeError as error:
        logger.error(
            "Invalid JSON in %s: %s",
            file_path,
            error
        )

    except OSError as error:
        logger.error(
            "Could not read %s: %s",
            file_path,
            error
        )

    return []


def preprocess_recipes(
    input_dir: Path = INPUT_DIR,
    output_file: Path = OUTPUT_FILE
) -> List[Dict[str, Any]]:
    """Process all recipe JSON files."""

    input_dir = Path(input_dir)
    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    recipes = []

    if not input_dir.exists():
        logger.error(
            "Recipe directory does not exist: %s",
            input_dir
        )
        return recipes

    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        logger.warning(
            "No JSON recipe files found in %s",
            input_dir
        )
        return recipes

    for file_path in json_files:

        raw_recipes = load_json_file(file_path)

        for raw_recipe in raw_recipes:

            if not isinstance(raw_recipe, dict):
                continue

            recipe = normalize_recipe(
                raw_recipe,
                file_path.name
            )

            if recipe:
                recipes.append(recipe)

    # Remove duplicate recipes
    unique_recipes = {}
    for recipe in recipes:
        unique_recipes[recipe["id"]] = recipe

    recipes = list(unique_recipes.values())

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            recipes,
            file,
            indent=2,
            ensure_ascii=False
        )

    logger.info(
        "Processed %d recipes",
        len(recipes)
    )

    logger.info(
        "Knowledge base saved to %s",
        output_file
    )

    return recipes


if __name__ == "__main__":
    preprocess_recipes()
