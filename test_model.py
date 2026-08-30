import torch
from Training.modeling import load_for_inference
from data_hand.db_tools.prompts import INSTRUCTION

print("Loading model...")
# To test the BASE model (to check if your LoRA weights are broken), change this to: load_for_inference(None)
processor, model = load_for_inference("./checkpoints/medgemma-qlora")

dialogue = """
Doctor: Hi there, what brings you in today?
Patient: I've had a really bad headache for the past 3 days. It's mostly right behind my eyes.
Doctor: Any sensitivity to light or nausea?
Patient: Yeah, the light really bothers me. No throwing up, but I feel a little sick to my stomach.
Doctor: Okay, it sounds like a migraine. Have you taken anything for it?
Patient: Just some Ibuprofen, but it didn't help much.
Doctor: I'll prescribe some Sumatriptan. Take one pill when you feel it starting.
"""

# 1. Build the content exactly like prompts.py does
content = INSTRUCTION["ACI"] + "\n\n" + dialogue

# 2. Put it in the chat format
messages = [{"role": "user", "content": content}]

# 3. CRITICAL: Apply the chat template so it adds <start_of_turn>user... etc.
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=prompt, return_tensors="pt").to(model.device)

print("\nGenerating medical note...\n" + "-"*30)
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=250,
        temperature=0.2,
        do_sample=True,
        # FIXED: Access the tokenizer's eos_token_id
        pad_token_id=processor.tokenizer.eos_token_id 
    )

# 4. Decode the generated tokens back into text
input_length = inputs["input_ids"].shape[1]
output_length = outputs[0].shape[0]

# --- DEBUGGING OUTPUT ---
# Print the raw, unsliced output including special tokens to see if it immediately stops
print("\n--- RAW MODEL OUTPUT (DEBUG) ---")
print(processor.decode(outputs[0], skip_special_tokens=False))
print("-" * 30)

print(f"Input tokens: {input_length}")
print(f"Total tokens output: {output_length}")
print(f"New tokens generated: {output_length - input_length}")
print("-" * 30)
# ------------------------

# Slice it so it only prints the NEW text
generated_tokens = outputs[0][input_length:]
generated_note = processor.decode(generated_tokens, skip_special_tokens=True)

print("\n--- FINAL NOTE ---")
print(generated_note)
print("-" * 30)
