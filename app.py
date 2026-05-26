import gradio as gr
from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

def summarize_article(article,max_len,min_len):

    if not article.strip():
        return "Please enter an article."

    summary=summarizer(
        article,
        max_length=max_len,
        min_length=min_len,
        do_sample=False
    )

    return summary[0]["summary_text"]

interface=gr.Interface(
    fn=summarize_article,
    inputs=[
        gr.Textbox(
            lines=15,
            placeholder="Paste article here...",
            label="Article"
        ),
        gr.Slider(50,300,value=120,step=10,label="Max Length"),
        gr.Slider(20,100,value=40,step=5,label="Min Length")
    ],
    outputs=gr.Textbox(label="Summary"),
    title="AI Article Summarizer",
    description="Summarize articles using Hugging Face Transformers"
)

if __name__=="__main__":
    interface.launch()