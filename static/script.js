let currentRecipe = null;
let currentStepIndex = 0;

async function fetchRecipes() {
  const res = await fetch("/api/recipes");
  const data = await res.json();
  displayRecipes(data.recipes);
}

async function searchRecipe() {
  const name = document.getElementById("recipeName").value.trim();
  if (!name) return alert("Enter a recipe name!");

  const res = await fetch("/api/cook", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ recipe_name: name })
  });

  const data = await res.json();

  if (!data.success) {
    document.getElementById("recipeDisplay").innerHTML = `<p>${data.message}</p>`;
    return;
  }

  currentRecipe = data;
  currentStepIndex = 0;
  showRecipe(data);
}

function displayRecipes(recipes) {
  let html = `<h2>Available Recipes</h2><ul>`;
  recipes.forEach(r => {
    html += `<li><b>${r.name_en || r.name}</b>: ${r.description}</li>`;
  });
  html += `</ul>`;
  document.getElementById("recipeDisplay").innerHTML = html;
}

function showRecipe(data) {
  let html = `
    <h2>🍲 ${data.recipe_name}</h2>
    <h4>Ingredients:</h4>
    <ul>${data.ingredients.map(i => `<li>${i}</li>`).join('')}</ul>
    <p><strong>Total steps:</strong> ${data.steps.length}</p>
    <p>Click "Start Cooking" to begin your step-by-step guide!</p>
  `;
  document.getElementById("recipeDisplay").innerHTML = html;
}

function startCooking() {
  if (!currentRecipe) {
    alert("Please search for a recipe first!");
    return;
  }
  currentStepIndex = 0;
  showCurrentStep();
}

function showCurrentStep() {
  const steps = currentRecipe.steps;
  if (currentStepIndex < steps.length) {
    document.getElementById("recipeDisplay").innerHTML = `
      <h2>Step ${currentStepIndex + 1} of ${steps.length}</h2>
      <p>${steps[currentStepIndex]}</p>
      <button id="nextStepBtn" class="blue">Next Step</button>
    `;
    document.getElementById("nextStepBtn").addEventListener("click", nextStep);
  } else {
    document.getElementById("recipeDisplay").innerHTML = `
      <h2>🎉 Cooking Complete!</h2>
      <p>Enjoy your delicious ${currentRecipe.recipe_name}!</p>
      <button id="showAll" class="orange">Show All Recipes</button>
    `;
    document.getElementById("showAll").addEventListener("click", fetchRecipes);
  }
}

function nextStep() {
  currentStepIndex++;
  showCurrentStep();
}

async function askAI() {
  const question = document.getElementById("aiQuestion").value.trim();
  if (!question) return alert("Ask something about cooking!");

  const recipeName = document.getElementById("recipeName").value.trim();
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      current_recipe: recipeName || null
    })
  });

  const data = await res.json();
  document.getElementById("aiResponse").innerHTML = `<p>${data.answer}</p>`;
}

document.getElementById("searchBtn").addEventListener("click", searchRecipe);
document.getElementById("showAll").addEventListener("click", fetchRecipes);
document.getElementById("startCooking").addEventListener("click", startCooking);
document.getElementById("debugInfo").addEventListener("click", () => {
  alert("Debug info: API running and frontend connected!");
});
document.getElementById("askBtn").addEventListener("click", askAI);
