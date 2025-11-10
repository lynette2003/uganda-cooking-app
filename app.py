from flask import Flask, request, jsonify
import os
import json
import glob
import random

app = Flask(__name__)

# ---------------------------
# Simple Cooking Assistant
# ---------------------------
class SimpleCookingAssistant:
    def __init__(self, recipe_folder_path: str):
        self.recipe_folder_path = recipe_folder_path
        self.recipes = {}
        self.current_recipe = None
        self.current_step = 0
        self.load_recipes_safe()
    
    def load_recipes_safe(self):
        """Load JSON recipes safely"""
        try:
            if not os.path.exists(self.recipe_folder_path):
                os.makedirs(self.recipe_folder_path)
                self.create_sample_recipe()
            
            json_files = glob.glob(os.path.join(self.recipe_folder_path, "*.json"))
            if not json_files:
                self.create_sample_recipe()
                json_files = glob.glob(os.path.join(self.recipe_folder_path, "*.json"))
            
            for file_path in json_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    recipe_name = data.get('name', os.path.splitext(os.path.basename(file_path))[0])
                    if isinstance(recipe_name, dict):
                        recipe_name = recipe_name.get('en', 'Unknown Recipe')
                    self.recipes[recipe_name.lower()] = {
                        'name': recipe_name,
                        'data': data,
                        'file_path': file_path
                    }
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        except Exception as e:
            print(f"Critical error loading recipes: {e}")
    
    def create_sample_recipe(self):
        """Create a sample recipe if none exist"""
        sample_recipe = {
            "name": "Ugandan Rolex",
            "description": "A popular Ugandan street food",
            "ingredients": {
                "main": [
                    {"name": "Chapati", "quantity": 1, "unit": "piece"},
                    {"name": "Eggs", "quantity": 2, "unit": "pieces"}
                ]
            },
            "steps": [
                {"instruction": "Chop vegetables"},
                {"instruction": "Cook eggs with vegetables"},
                {"instruction": "Roll in chapati and serve"}
            ]
        }
        sample_path = os.path.join(self.recipe_folder_path, "sample_rolex.json")
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample_recipe, f, indent=2)
    
    # ---------------------------
    # Helper functions
    # ---------------------------
    def get_ingredients(self, recipe_data):
        ingredients = []
        ing_data = recipe_data.get('ingredients', {})
        if isinstance(ing_data, dict):
            for category, items in ing_data.items():
                for item in items:
                    qty = item.get('quantity', '')
                    unit = item.get('unit', '')
                    ingredients.append(f"{qty} {unit} {item['name']}".strip())
        elif isinstance(ing_data, list):
            for item in ing_data:
                qty = item.get('quantity', '')
                unit = item.get('unit', '')
                ingredients.append(f"{qty} {unit} {item['name']}".strip())
        return ingredients if ingredients else ["Traditional ingredients"]

    def get_steps(self, recipe_data):
        steps = []
        for step in recipe_data.get('steps', []):
            if isinstance(step, dict) and 'instruction' in step:
                steps.append(step['instruction'])
        return steps if steps else ["Prepare ingredients", "Cook using traditional methods", "Serve hot"]

    def answer_cooking_question(self, question, current_recipe=None):
        q = question.lower()
        if current_recipe:
            recipe_name = current_recipe['name']
            recipe_data = current_recipe['data']
            if any(word in q for word in ['ingredient', 'what is in', 'what goes in', 'contains']):
                ing = self.get_ingredients(recipe_data)
                return f"For {recipe_name}, you'll need: {', '.join(ing[:5])}..." if len(ing)>5 else f"For {recipe_name}, you'll need: {', '.join(ing)}"
            if any(word in q for word in ['how long', 'time', 'minutes', 'hours']):
                prep_time = recipe_data.get('prep_time_mins', 'Unknown')
                cook_time = recipe_data.get('cook_time_mins', 'Unknown')
                return f"{recipe_name} takes {prep_time} mins to prepare and {cook_time} mins to cook."
            if any(word in q for word in ['how to make', 'how to cook', 'steps']):
                steps = self.get_steps(recipe_data)
                return f"To make {recipe_name}, follow these {len(steps)} steps. Click 'Start Cooking' to see them step by step!"
        
        fallback = [
            "I can help with Ugandan cooking techniques, ingredient substitutions, and traditional methods.",
            "Ask about specific Ugandan ingredients or recipes!",
            "I specialize in Ugandan cuisine - from street foods like Rolex to traditional dishes like Luwombo."
        ]
        return random.choice(fallback)

# Initialize assistant
assistant = SimpleCookingAssistant('data/recipes')

# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def home():
    return "<h1>Ugandan Cooking Assistant API</h1><p>Use /api/recipes to get started!</p>"

@app.route('/api/recipes', methods=['GET'])
def get_recipes():
    recipes = []
    for r in assistant.recipes.values():
        data = r['data']
        desc = data.get('description', 'Traditional recipe')
        recipes.append({'name': r['name'], 'description': desc})
    return jsonify({'success': True, 'recipes': recipes})

@app.route('/api/search', methods=['POST'])
def search_recipes():
    keyword = request.json.get('keyword', '').lower()
    matches = [r['name'] for k,r in assistant.recipes.items() if keyword in k or keyword in r['name'].lower()]
    return jsonify({'matches': matches})

@app.route('/api/cook', methods=['POST'])
def cook_recipe():
    recipe_name = request.json.get('recipe_name', '').lower()
    recipe = assistant.recipes.get(recipe_name)
    if not recipe:
        return jsonify({'success': False, 'message': 'Recipe not found'})
    assistant.current_recipe = recipe
    assistant.current_step = 0
    return jsonify({
        'success': True,
        'recipe_name': recipe['name'],
        'ingredients': assistant.get_ingredients(recipe['data']),
        'steps': assistant.get_steps(recipe['data']),
        'total_steps': len(assistant.get_steps(recipe['data']))
    })

@app.route('/api/next', methods=['POST'])
def next_step():
    if not assistant.current_recipe:
        return jsonify({'success': False, 'message': 'No active recipe'})
    steps = assistant.get_steps(assistant.current_recipe['data'])
    if assistant.current_step >= len(steps):
        name = assistant.current_recipe['name']
        assistant.current_recipe = None
        assistant.current_step = 0
        return jsonify({'success': True, 'completed': True, 'message': f'You cooked {name}!'})
    step_text = steps[assistant.current_step]
    assistant.current_step += 1
    return jsonify({'success': True, 'current_step': assistant.current_step, 'total_steps': len(steps), 'step_text': step_text})

@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')
    current = data.get('current_recipe')
    current_recipe = assistant.recipes.get(current.lower()) if current else None
    answer = assistant.answer_cooking_question(question, current_recipe)
    return jsonify({'success': True, 'answer': answer})

# ---------------------------
# Run app
# ---------------------------
if __name__ == '__main__':
    print("🍳 Ugandan Cooking Assistant API starting...")
    app.run(host='0.0.0.0', port=5000, debug=True)
