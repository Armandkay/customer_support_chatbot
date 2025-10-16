import gradio as gr
from transformers import T5Tokenizer, TFT5ForConditionalGeneration

MODEL_DIR = "./checkpoints"
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = TFT5ForConditionalGeneration.from_pretrained(MODEL_DIR)

def respond(message):
    input_text = "User: " + message + " Agent:"
    inputs = tokenizer(input_text, return_tensors="tf", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=40)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

iface = gr.Interface(fn=respond, inputs="text", outputs="text", title="Customer Support Chatbot")
iface.launch()
