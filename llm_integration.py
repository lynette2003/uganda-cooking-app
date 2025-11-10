import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging
import random

logger = logging.getLogger(__name__)

class CookingGPT2:
    def __init__(self, use_pre_trained: bool = True):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        if use_pre_trained:
            self.load_pre_trained_model()
    
    def load_pre_trained_model(self):
        """Load pre-trained GPT-2 model"""
        try:
            logger.info("Loading pre-trained GPT-2 model...")
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            
            # Add padding token
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model_loaded = True
            logger.info("Pre-trained GPT-2 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load GPT-2 model: {e}")
            self.model_loaded = False
    
    def generate_response(self, prompt: str, max_length: int = 150, temperature: float = 0.7) -> str:
        """Generate response using GPT-2 model"""
        if not self.model_loaded:
            return None
        
        try:
            # Enhanced cooking-specific prompt
            cooking_prompt = f"""As an expert Ugandan cooking assistant, provide helpful and accurate cooking advice.

Question: {prompt}

Answer:"""
            
            # Tokenize input
            inputs = self.tokenizer.encode(cooking_prompt, return_tensors='pt', max_length=512, truncation=True)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    num_return_sequences=1,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    early_stopping=True
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the answer part
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()
            
            # Clean up response
            response = response.replace(cooking_prompt, '').strip()
            
            return response if len(response) > 10 else None
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None
    
    def answer_cooking_question(self, question: str, context: str = "") -> str:
        """Generate cooking-specific response with enhanced prompting"""
        if not self.model_loaded:
            return None
        
        # Enhanced context for better responses
        enhanced_context = f"""
        You are a Ugandan cooking expert. Provide practical, culturally appropriate cooking advice.
        Focus on Ugandan cuisine, local ingredients, and traditional cooking methods.
        Be specific and helpful.
        
        Context: {context}
        Question: {question}
        """
        
        response = self.generate_response(enhanced_context)
        return response