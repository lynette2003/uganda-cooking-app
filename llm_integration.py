"""
llm_integration.py

RAG-based Ugandan Cooking Assistant.

Pipeline:

User question
↓
Sentence Transformer embedding
↓
Semantic recipe retrieval
↓
Relevant recipe context
↓
Answer built directly from retrieved recipes
   (optional GPT-2 rewrite, off by default)

The recipe knowledge base is the source of truth.

GPT-2 (the small, non-instruction-tuned base model) does not follow
instructions reliably and can hallucinate ingredients, steps, and
even sources when used for free-text generation. Because the
knowledge base already contains clean, structured recipes, the
default behaviour is to build the answer directly from the
retrieved recipes rather than asking GPT-2 to generate free text.
GPT-2 generation is still available (see `use_llm_generation`) for
experimentation, but it is opt-in and not recommended for
production use with this model.
"""

# ============================================================
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

from transformers import GPT2LMHeadModel, GPT2Tokenizer


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# ============================================================
# PATHS AND MODELS
# ============================================================

KNOWLEDGE_BASE = Path(
    "data/processed/recipe_knowledge_base.json"
)

EMBEDDINGS_FILE = Path(
    "data/processed/recipe_embeddings.npy"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

DEFAULT_LLM = "gpt2"

# ============================================================
# RAG CLASS
# ============================================================

class UgandanCookingRAG:

    def __init__(
        self,
        llm_model_name: str = DEFAULT_LLM,
        embedding_model_name: str = EMBEDDING_MODEL,
        load_llm: bool = False,
        use_llm_generation: bool = False
    ):
        """
        load_llm:
            Whether to load the GPT-2 model at all. Defaults to
            False because the default answering strategy does not
            need it.

        use_llm_generation:
            Whether answer_cooking_question() should actually use
            GPT-2 to generate the answer text. Defaults to False:
            answers are built directly from the retrieved recipes,
            which is faster and does not risk hallucinated content.
            Set this to True only if you specifically want to
            experiment with free-text generation, and note that
            load_llm must also be True (or set automatically, see
            below) for this to have any effect.
        """

        self.embedding_model = None
        self.llm_model = None
        self.tokenizer = None

        self.recipes: List[Dict] = []
        self.recipe_embeddings = None

        self.llm_loaded = False
        self.use_llm_generation = use_llm_generation

        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name

        self._load_knowledge_base()
        self._load_embeddings()
        self._load_embedding_model()

        # If the caller asked for LLM generation but didn't
        # explicitly request loading the model, load it anyway -
        # otherwise use_llm_generation would silently do nothing.
        if load_llm or use_llm_generation:
            self._load_llm()

    # ========================================================
    # KNOWLEDGE BASE
    # ========================================================

    def _load_knowledge_base(self):

        if not KNOWLEDGE_BASE.exists():
            raise FileNotFoundError(
                f"Knowledge base not found: "
                f"{KNOWLEDGE_BASE}. "
                f"Run data_preprocessing.py first."
            )

        with open(
            KNOWLEDGE_BASE,
            "r",
            encoding="utf-8"
        ) as file:

            self.recipes = json.load(file)

        if not isinstance(self.recipes, list):
            raise ValueError(
                "Recipe knowledge base must contain "
                "a list of recipes."
            )

        logger.info(
            "Loaded %d recipes.",
            len(self.recipes)
        )

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    def _load_embeddings(self):

        if not EMBEDDINGS_FILE.exists():
            raise FileNotFoundError(
                f"Recipe embeddings not found: "
                f"{EMBEDDINGS_FILE}. "
                f"Run train_llm.py first."
            )

        self.recipe_embeddings = np.load(
            EMBEDDINGS_FILE
        )

        if len(self.recipe_embeddings) != len(self.recipes):
            raise ValueError(
                "The number of recipe embeddings does not "
                "match the number of recipes in the "
                "knowledge base."
            )

        logger.info(
            "Loaded recipe embeddings: %s",
            self.recipe_embeddings.shape
        )

    # ========================================================
    # EMBEDDING MODEL
    # ========================================================

    def _load_embedding_model(self):

        logger.info(
            "Loading embedding model: %s",
            self.embedding_model_name
        )

        self.embedding_model = SentenceTransformer(
            self.embedding_model_name
        )

        logger.info(
            "Embedding model loaded successfully."
        )

    # ========================================================
    # GPT-2
    # ========================================================

    def _load_llm(self):

        try:

            logger.info(
                "Loading language model: %s",
                self.llm_model_name
            )

            self.tokenizer = GPT2Tokenizer.from_pretrained(
                self.llm_model_name
            )

            self.llm_model = GPT2LMHeadModel.from_pretrained(
                self.llm_model_name
            )

            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

            self.llm_loaded = True

            logger.info(
                "Language model loaded successfully."
            )

        except Exception as error:

            logger.error(
                "Could not load language model: %s",
                error
            )

            self.llm_loaded = False

    # ========================================================
    # RETRIEVAL
    # ========================================================

    def retrieve_recipes(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.25
    ) -> List[Dict]:
        """
        Retrieve recipes semantically related to a question.
        """

        if not query or not query.strip():
            return []

        query = query.strip()

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        corpus_embeddings = torch.tensor(
            self.recipe_embeddings,
            dtype=torch.float32
        )

        hits = semantic_search(
            query_embedding,
            corpus_embeddings,
            top_k=min(
                top_k,
                len(self.recipes)
            )
        )[0]

        results = []

        for hit in hits:

            score = float(
                hit["score"]
            )

            if score < min_score:
                continue

            recipe_index = hit["corpus_id"]

            recipe = dict(
                self.recipes[recipe_index]
            )

            recipe["similarity_score"] = round(
                score,
                4
            )

            results.append(recipe)

        logger.info(
            "Retrieved %d relevant recipes for query: %s",
            len(results),
            query
        )

        return results

    # ========================================================
    # CONTEXT CREATION
    # ========================================================

    def _format_recipe_context(self, recipes: List[Dict]) -> str:

        if not recipes:
            return (
                "No matching recipes were found "
                "in the Ugandan recipe knowledge base."
            )

        context_parts = []

        for index, recipe in enumerate(recipes, start=1):

            name = recipe.get("name", "")

            local_name = ""

            if isinstance(name, dict):
                local_name = name.get("local_name", "")
                name = name.get("en", local_name)

            elif not isinstance(name, str):
                name = str(name)

            if not name:
                name = "Unnamed Recipe"

            description = recipe.get("description", "")

            if isinstance(description, dict):
                description = description.get("en", "")

            ingredients = recipe.get("ingredients", [])

            ingredient_lines = []

            if isinstance(ingredients, list):

                for ingredient in ingredients:

                    if isinstance(ingredient, dict):

                        ingredient_name = ingredient.get(
                            "name",
                            ingredient.get("ingredient", "")
                        )

                        quantity = ingredient.get(
                            "quantity",
                            ""
                        )

                        unit = ingredient.get(
                            "unit",
                            ""
                        )

                        ingredient_text = str(
                            ingredient_name
                        )

                        if quantity:
                            ingredient_text += (
                                f": {quantity}"
                            )

                        if unit:
                            ingredient_text += (
                                f" {unit}"
                            )

                    else:
                        ingredient_text = str(
                            ingredient
                        )

                    if ingredient_text.strip():
                        ingredient_lines.append(
                            f"- {ingredient_text}"
                        )

            else:

                ingredient_lines.append(
                    f"- {ingredients}"
                )

            ingredients_text = "\n".join(
                ingredient_lines
            )

            instructions = recipe.get(
                "instructions",
                []
            )

            instruction_lines = []

            if isinstance(instructions, list):

                for i, step in enumerate(
                    instructions,
                    start=1
                ):

                    if isinstance(step, dict):

                        instruction = step.get(
                            "instruction",
                            ""
                        )

                    else:

                        instruction = str(step)

                    if instruction:
                        instruction_lines.append(
                            f"{i}. {instruction}"
                        )

            elif isinstance(instructions, str):

                instruction_lines.append(
                    instructions
                )

            instructions_text = "\n".join(
                instruction_lines
            )

            cooking_method = recipe.get(
                "cooking_method",
                ""
            )

            meal_type = recipe.get(
                "meal_type",
                ""
            )

            region = recipe.get(
                "region",
                ""
            )

            dietary_information = recipe.get(
                "dietary_information",
                ""
            )

            context = f"""RECIPE {index}

Name:
{name}

Local name:
{local_name}

Description:
{description}

Ingredients:
{ingredients_text}

Instructions:
{instructions_text}

Cooking method:
{cooking_method}

Meal type:
{meal_type}

Region:
{region}

Dietary information:
{dietary_information}"""

            context_parts.append(
                context.strip()
            )

        return "\n\n".join(
            context_parts
        )

    # ========================================================
    # PROMPT
    # ========================================================

    def _create_prompt(
        self,
        question: str,
        recipes: List[Dict]
    ) -> str:

        context = self._format_recipe_context(
            recipes
        )

        return f"""You are a helpful Ugandan cooking assistant.

Use the recipe information provided below as your
main source of information.

IMPORTANT RULES:

1. Prefer information from the provided recipes.
2. Do not invent ingredients or cooking steps.
3. If the recipes do not contain enough information,
   clearly say so.
4. Keep the answer practical and easy to understand.
5. Respect Ugandan cooking traditions and local ingredients.
6. When discussing substitutions, clearly label them
   as suggestions rather than facts from the recipe.

RECIPE KNOWLEDGE BASE:

{context}

USER QUESTION:

{question}

ANSWER:
""".strip()

    # ========================================================
    # GENERATION
    # ========================================================

    def _generate_with_gpt2(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.7
    ) -> Optional[str]:

        if not self.llm_loaded:
            return None

        try:

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=800
            )

            with torch.no_grad():

                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                    pad_token_id=(
                        self.tokenizer.eos_token_id
                    )
                )

            generated_text = (
                self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
            )

            if "ANSWER:" in generated_text:

                generated_text = (
                    generated_text
                    .split("ANSWER:", 1)[1]
                    .strip()
                )

            return generated_text.strip()

        except Exception as error:

            logger.error(
                "Generation failed: %s",
                error
            )

            return None

    # ========================================================
    # RETRIEVAL-BASED ANSWER (default, no LLM generation)
    # ========================================================

    @staticmethod
    def _get_recipe_name(recipe: Dict) -> str:

        name = recipe.get("name", "")

        if isinstance(name, dict):
            name = name.get("en") or name.get("local_name") or ""

        elif not isinstance(name, str):
            name = str(name)

        return name.strip() or "Unnamed Recipe"

    @staticmethod
    def _format_ingredient_line(ingredient) -> str:

        if isinstance(ingredient, dict):

            name = ingredient.get(
                "name",
                ingredient.get("ingredient", "")
            )

            quantity = ingredient.get("quantity", "")
            unit = ingredient.get("unit", "")

            text = str(name)

            if quantity:
                text += f": {quantity}"

            if unit:
                text += f" {unit}"

            return text.strip()

        return str(ingredient).strip()

    @staticmethod
    def _format_instruction_line(step) -> str:

        if isinstance(step, dict):
            return str(step.get("instruction", "")).strip()

        return str(step).strip()

    def _format_single_recipe_answer(self, recipe: Dict) -> str:
        """
        Build a clean, human-readable answer for one recipe,
        using only the fields present in the knowledge base.
        """

        recipe_name = self._get_recipe_name(recipe)

        description = recipe.get("description", "")

        if isinstance(description, dict):
            description = description.get("en", "")

        ingredients = recipe.get("ingredients", [])
        instructions = recipe.get("instructions", [])

        lines = [f"**{recipe_name}**"]

        if description:
            lines.append(str(description).strip())

        if ingredients:

            lines.append("\nIngredients:")

            for ingredient in ingredients:

                text = self._format_ingredient_line(ingredient)

                if text:
                    lines.append(f"- {text}")

        if instructions:

            lines.append("\nInstructions:")

            for index, step in enumerate(instructions, start=1):

                text = self._format_instruction_line(step)

                if text:
                    lines.append(f"{index}. {text}")

        if not ingredients and not instructions:

            lines.append(
                "\nThe knowledge base doesn't have full "
                "ingredients or steps for this recipe yet."
            )

        return "\n".join(lines).strip()

    def _create_answer_from_recipes(
        self,
        question: str,
        recipes: List[Dict]
    ) -> str:
        """
        Build the answer directly from the retrieved recipes,
        without any free-text LLM generation. This is the default
        way answers are produced, since it guarantees the answer
        only contains ingredients and steps that actually exist
        in the knowledge base.
        """

        if not recipes:

            return (
                "I couldn't find a matching recipe in the "
                "Ugandan cooking knowledge base. Try asking "
                "about a specific dish (e.g. 'How do I make "
                "matooke?') or a specific ingredient."
            )

        if len(recipes) == 1:

            intro = "Here's what I found in the recipe knowledge base:\n\n"

            return intro + self._format_single_recipe_answer(
                recipes[0]
            )

        intro = (
            "I found a few recipes in the knowledge base that "
            "might match what you're looking for:\n\n"
        )

        sections = [
            self._format_single_recipe_answer(recipe)
            for recipe in recipes
        ]

        return intro + "\n\n---\n\n".join(sections)

    # ========================================================
    # MAIN RAG FUNCTION
    # ========================================================

    def answer_cooking_question(
        self,
        question: str,
        top_k: int = 3
    ) -> Dict:

        if not question or not question.strip():

            return {
                "answer": (
                    "Please ask me a cooking question."
                ),
                "recipes": []
            }

        retrieved_recipes = self.retrieve_recipes(
            question,
            top_k=top_k
        )

        answer = None

        if self.use_llm_generation and self.llm_loaded:

            prompt = self._create_prompt(
                question,
                retrieved_recipes
            )

            answer = self._generate_with_gpt2(
                prompt
            )

        if not answer:

            answer = self._create_answer_from_recipes(
                question,
                retrieved_recipes
            )

        return {
            "answer": answer,
            "recipes": [
                {
                    "id": recipe.get("id"),
                    "name": recipe.get("name"),
                    "similarity_score": recipe.get(
                        "similarity_score"
                    )
                }
                for recipe in retrieved_recipes
            ]
        }


# ============================================================
# SINGLE RAG INSTANCE
# ============================================================

_rag_instance = None


def get_cooking_assistant():
    """
    Return a single shared RAG assistant instance.
    """

    global _rag_instance

    if _rag_instance is None:
        _rag_instance = UgandanCookingRAG()

    return _rag_instance


# ============================================================
# BACKWARDS-COMPATIBLE API
# ============================================================

def answer_cooking_question(
    question: str,
    context: str = ""
) -> Optional[str]:
    """
    Backwards-compatible helper.

    Existing code can call:

        answer_cooking_question(question)
    """

    assistant = get_cooking_assistant()

    result = assistant.answer_cooking_question(
        question
    )

    return result["answer"]


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    # Default: answers come straight from the retrieved recipes,
    # no GPT-2 loaded or used. Pass use_llm_generation=True to
    # experiment with free-text GPT-2 generation instead.
    assistant = UgandanCookingRAG()

    question = "How do I prepare matooke?"

    result = assistant.answer_cooking_question(
        question
    )

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(result["answer"])

    print("\nRETRIEVED RECIPES:")

    for recipe in result["recipes"]:

        print(
            f"- {recipe['name']} "
            f"({recipe['similarity_score']})"
        )

    print("\nMethods in UgandanCookingRAG:")
    print([
        method
        for method in dir(UgandanCookingRAG)
        if "answer" in method.lower()
    ])