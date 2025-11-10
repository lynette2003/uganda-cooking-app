
# quick_finetune.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import os

def create_training_data(recipes_folder: str, output_file: str):
    """Create training data from your recipes"""
    training_texts = []
    
    # Load all recipes and create training examples
    for file_name in os.listdir(recipes_folder):
        if file_name.endswith('.json'):
            file_path = os.path.join(recipes_folder, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                import json
                recipe_data = json.load(f)
                
                # Create training examples
                recipe_name = file_name.replace('.json', '').replace('_', ' ').title()
                
                # Example training formats
                training_texts.append(f"Question: How do I make {recipe_name}? Answer: ")
                training_texts.append(f"Question: What ingredients do I need for {recipe_name}? Answer: ")
                training_texts.append(f"Question: Can you explain {recipe_name}? Answer: ")
                training_texts.append(f"Question: What can I substitute in {recipe_name}? Answer: ")
    
    # Save training data
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(training_texts))

def quick_finetune():
    """Quick fine-tune GPT-2 on cooking data"""
    
    # Load pre-trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Create training data
    create_training_data('data/recipes', 'training_data.txt')
    
    # Prepare dataset
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path="training_data.txt",
        block_size=128
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./cooking-gpt2',
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained('./cooking-gpt2')

if __name__ == '__main__':
    quick_finetune()