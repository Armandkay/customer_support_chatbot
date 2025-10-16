from transformers import T5Tokenizer, TFT5ForConditionalGeneration

MODEL_DIR = "./checkpoints"
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = TFT5ForConditionalGeneration.from_pretrained(MODEL_DIR)

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        break
    input_text = "User: " + user_input + " Agent:"
    inputs = tokenizer(input_text, return_tensors="tf", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=40)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Bot:", response)
