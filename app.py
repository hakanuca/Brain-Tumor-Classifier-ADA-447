# --------------------
# HAKAN UCA
# ADA - 447
# MIDTERM PROJECT
# app.py
# --------------------


from fastai.vision.all import *
import gradio as gr
import os

# Trained model load
learn = load_learner("tumormodelfinal.pkl")
labels = learn.dls.vocab

# Prediction function
def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

# Frontend tarafı (renkler generic fontlar değiştirilebilir)
title_html = """
<h1 style='text-align: center; color: white; font-size: 48px;'>🧠 Brain Tumor Classifier</h1>
<h3 style='text-align: center; color: #FDFEFE; font-size: 26px;'>Prepared by Hakan Uca | ADA-447 | Midterm Project</h3>
"""

description_md = """
<div style='font-size: 18px;'>
<h3>🧬 About this System:</h3>
<p>This model is designed to automatically detect brain tumors from MRI images. It learns to recognize the presence of a tumor by analyzing many labeled MRI scans (tumor or no tumor).</p>
<p>After training, the model can predict whether a tumor is present or not in a new MRI image. This helps doctors with preliminary diagnoses and allows for faster and more accurate analysis.</p>

<strong>Instructions:</strong>
<ul>
    <li>Click "Upload" to use your own image.</li>
    <li>Or click a sample image to try it out.</li>
    <li>The model returns a classification: <strong>Yes</strong> (tumor) or <strong>No</strong> (no tumor), along with probabilities.</li>
</ul>
<hr>
</div>
"""

# Örnek görseller
example_images = [
    "examples/no1.jpeg",
    "examples/no2.jpeg",
    "examples/no3.jpg",
    "examples/yes1.JPG",
    "examples/yes2.jpg",
    "examples/yes3.JPG",
    "examples/yes4.jpg"
]

# Arayüz blokları
with gr.Blocks(css="""
body, .gr-block, .gr-box, .gr-row, .gr-column, .gr-form {
    font-size: 18px !important;
}

/* Classify butonu özel rengi */
#classify-btn button {
    background-color: #28a745 !important; /* Yeşil */
    color: white !important;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 20px;
}
""") as app:
    gr.HTML(title_html)
    gr.HTML(description_md)

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Brain Scan")
        output = gr.Label(num_top_classes=2, label="Prediction")

    gr.Examples(examples=example_images, inputs=image_input, label="Sample Images")

    submit_btn = gr.Button("🔍 Classify", elem_id="classify-btn")
    submit_btn.click(fn=classify_image, inputs=image_input, outputs=output)

    gr.HTML("""
    <hr>
    <p style='text-align: center; font-size: 16px; color: #ABB2B9;'>
    ⚠️ This is an AI-based system trained on a dataset and may not always reflect the true medical condition. Always consult a healthcare professional for a definitive diagnosis.
    </p>
    <br>
    <div style='text-align: center; display: flex; justify-content: center; gap: 20px; margin-bottom: 20px;'>
        <a href="https://github.com/HakanUca" target="_blank">
            <button style="
                background-color: #6e5494;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 16px;
                border-radius: 6px;
                cursor: pointer;
            ">🐙 GitHub</button>
        </a>
        <a href="https://www.linkedin.com/in/hakanuca/" target="_blank">
            <button style="
                background-color: #4078c0;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 16px;
                border-radius: 6px;
                cursor: pointer;
            ">💼 LinkedIn</button>
        </a>
    </div>
    """)

# Uygulamayı başlat
app.launch()
