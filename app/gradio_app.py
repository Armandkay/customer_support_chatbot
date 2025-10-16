import gradio as gr
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
import tensorflow as tf
import traceback

# Load model and tokenizer
model_path = "./checkpoints"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = TFT5ForConditionalGeneration.from_pretrained(model_path)

def chat(message: str, history: list = []):
    try:
        input_ids = tokenizer.encode(message, return_tensors="tf")
        output = model.generate(input_ids, max_length=100, num_beams=2)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print("❌ ERROR in chat():", str(e))
        traceback.print_exc()
        return "Sorry, I ran into an internal error. Check the terminal for details."

demo = gr.ChatInterface(
    fn=chat,
    title="💬 Customer Support Chatbot",
    description="Chat with your fine-tuned T5 model trained on support data.",
    type="messages"
)

demo.launch()
