# import gradio as gr

# gr.Interface.load("models/Abijith/Text-summarizer-t5-small").launch()

import os
import gradio as gr
from transformers import AutoTokenizer, T5ForConditionalGeneration

model_name = 'Abijith/Billsum-text-summarizer-t5-small'
# Load model directly
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def summarize_text(input_text):
    summar_input = 'summarize: '+input_text
    input_tokens = tokenizer(summar_input, return_tensors='pt').input_ids
    outputs = model.generate(input_tokens, max_new_tokens=100, min_new_tokens=30, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Interface for the Gradio app
iface = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(lines=5, label="Input Text"),
    outputs=gr.Textbox(label="Summary"),
    title="Text Summarizer",
    description="Enter a paragraph, and the app will provide a summary.",
)

# Launch the Gradio app
iface.launch()
    