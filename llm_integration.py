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
GPT-2 response generation

The recipe knowledge base is the source of truth.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class UgandanCookingRAG:

    def __init__(
        self,
        llm_model_name: str = DEFAULT_LLM,
        embedding_model_name: str = EMBEDDING_MODEL,
        load_llm: bool = True
    ):
        self.embedding_model = None

        self.llm_model = None
        self.tokenizer = None

        self.recipes: List[Dict] = []
        self.recipe_embeddings = None

        self.llm_loaded = False

        self.embedding_model_name = (
            embedding_model_name
        )

        self.llm_model_name = llm_model_name

        self._load_knowledge_base()
        self._load_embeddings()
        self._load_embedding_model()

        if load_llm:
            self._load_llm()

    # --------------------------------------------------
    # KNOWLEDGE BASE
    # --------------------------------------------------

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

        logger.info(
            "Loaded %d recipes.",
            len(self.recipes)
        )

    # --------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------

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

        logger.info(
            "Loaded recipe embeddings: %s",
            self.recipe_embeddings.shape
        )

    # --------------------------------------------------
    # EMBEDDING MODEL
    # --------------------------------------------------

    def _load_embedding_model(self):

        logger.info(
            "Loading embedding model..."
        )

        self.embedding_model = SentenceTransformer(
            self.embedding_model_name
        )

        logger.info(
            "Embedding model loaded."
        )

    # --------------------------------------------------
    # GPT-2
    # --------------------------------------------------

    def _load_llm(self):

        try:

            logger.info(
                "Loading language model: %s",
                self.llm_model_name
            )

            self.tokenizer = (
                GPT2Tokenizer.from_pretrained(
                    self.llm_model_name
                )
            )

            self.llm_model = (
                GPT2LMHeadModel.from_pretrained(
                    self.llm_model_name
                )
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

    # --------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------

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

        query_embedding = (
            self.embedding_model.encode(
                [query],
                convert_to_tensor=True,
                normalize_embeddings=True
            )
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

            score = float(hit["score"])

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

    # --------------------------------------------------
    # CONTEXT CREATION
    # --------------------------------------------------

    def _format_recipe_context(
        self,
        recipes: List[Dict]
    ) -> str:

        if not recipes:
            return (
                "No matching recipes were found "
                "in the Ugandan recipe knowledge base."
            )

        context_parts = []

        for index, recipe in enumerate(
            recipes,
            start=1
        ):

            ingredients = "\n".join(
                f"- {ingredient}"
                for ingredient
                in recipe.get(
                    "ingredients",
                    []
                )
            )

            instructions = "\n".join(
                f"{i + 1}. {step}"
                for i, step
                in enumerate(
                    recipe.get(
                        "instructions",
                        []
                    )
                )
            )

            context = f"""
RECIPE {index}
Name: {recipe.get('name', '')}

Description:
{recipe.get('description', '')}

Ingredients:
{ingredients}

Instructions:
{instructions}

Cooking method:
{recipe.get('cooking_method', '')}

Meal type:
{recipe.get('meal_type', '')}

Region:
{recipe.get('region', '')}

Dietary information:
{recipe.get('dietary_information', '')}
""".strip()

            context_parts.append(context)

        return "\n\n".join(context_parts)

    # --------------------------------------------------
    # PROMPT
    # --------------------------------------------------

    def _create_prompt(
        self,
        question: str,
        recipes: List[Dict]
    ) -> str:

        context = self._format_recipe_context(
            recipes
        )

        return f"""
You are a helpful Ugandan cooking assistant.

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

    # --------------------------------------------------
    # GENERATION
    # --------------------------------------------------

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
                max_length=900
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

            # Return only text generated after ANSWER:
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

    # --------------------------------------------------
    # FALLBACK ANSWER
    # --------------------------------------------------

    def _create_fallback_answer(
        self,
        question: str,
        recipes: List[Dict]
    ) -> str:

        if not recipes:

            return (
                "I couldn't find a matching recipe in "
                "the Ugandan cooking knowledge base. "
                "Try asking about a specific dish or "
                "ingredient."
            )

        recipe = recipes[0]

        ingredients = recipe.get(
            "ingredients",
            []
        )

        instructions = recipe.get(
            "instructions",
            []
        )

        answer = (
            f"Based on the recipe knowledge base, "
            f"you could make {recipe['name']}.\n\n"
        )

        if ingredients:

            answer += "Ingredients:\n"

            for ingredient in ingredients:

                answer += (
                    f"• {ingredient}\n"
                )

            answer += "\n"

        if instructions:

            answer += "Preparation:\n"

            for index, step in enumerate(
                instructions,
                start=1
            ):

                answer += (
                    f"{index}. {step}\n"
                )

        return answer.strip()

    # --------------------------------------------------
    # MAIN RAG FUNCTION
    # --------------------------------------------------

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

        retrieved_recipes = (
            self.retrieve_recipes(
                question,
                top_k=top_k
            )
        )

        prompt = self._create_prompt(
            question,
            retrieved_recipes
        )

        answer = self._generate_with_gpt2(
            prompt
        )

        # GPT-2 fallback
        if not answer:

            answer = self._create_fallback_answer(
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


# ------------------------------------------------------
# SIMPLE API FOR YOUR EXISTING APP
# ------------------------------------------------------

_rag_instance = None


def get_cooking_assistant():

    global _rag_instance

    if _rag_instance is None:

        _rag_instance = UgandanCookingRAG()

    return _rag_instance


def answer_cooking_question(
    question: str,
    context: str = ""
) -> Optional[str]:

    """
    Backwards-compatible function.

    Existing code can continue calling:

        answer_cooking_question(question)
    """

    assistant = get_cooking_assistant()

    result = assistant.answer_cooking_question(
        question
    )

    return result["answer"]


# ------------------------------------------------------
# TEST
# ------------------------------------------------------

if __name__ == "__main__":

    assistant = UgandanCookingRAG()

    question = (
        "How do I prepare matooke?"
    )

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
