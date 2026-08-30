from Training.modeling import load_for_inference

# 1. Load your trained model and processor
processor, model = load_for_inference("checkpoints/medgemma-qlora")

# 2. Define your target Hugging Face repository name
# Replace "your-username" with your actual HF username
repo_id = "Tser-vak/medgemma-4b-medical-notes"

# 3. Push the model adapter weights
model.push_to_hub(repo_id)

# 4. Push the processor (so others can tokenize inputs exactly the same way)
processor.push_to_hub(repo_id)

print(f"Successfully pushed to https://huggingface.co/{repo_id}")
