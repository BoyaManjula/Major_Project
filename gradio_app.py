import os
import datetime
import shutil
import gradio as gr
from PIL import Image
from dotenv import load_dotenv

from image_analyzer import analyze_image
from doctor_brain import analyze_text_query
from doctor_voice import text_to_speech
from doctor_voice_stt import transcribe_with_groq
from report_analyzer import analyze_report

load_dotenv()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# -------------------------
# Save uploaded image
# -------------------------
def save_image(image_numpy):

    filename = f"image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    path = os.path.join(UPLOAD_DIR, filename)

    img = Image.fromarray(image_numpy)
    img.save(path)

    return path


# -------------------------
# Save uploaded report (FIXED)
# -------------------------
def save_report(file):

    if file is None:
        return None

    filename = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    path = os.path.join(UPLOAD_DIR, filename)

    # ✅ FIX: file is already a path → just copy it
    shutil.copy(file, path)

    return path


# -------------------------
# MAIN PROCESS FUNCTION
# -------------------------
def process_inputs(text_input, audio_input, image_input, report_input):

    medical_context = ""

    # -------------------------
    # IMAGE INPUT
    # -------------------------
    if image_input is not None:

        image_path = save_image(image_input)

        image_result = analyze_image(image_path)

        medical_context += f"Image Analysis Result: {image_result}\n"


    # -------------------------
    # VOICE INPUT
    # -------------------------
    if audio_input and os.path.exists(audio_input):

        voice_text = transcribe_with_groq(audio_input)

        medical_context += f"User Voice Input: {voice_text}\n"


    # -------------------------
    # TEXT INPUT
    # -------------------------
    if text_input and text_input.strip():

        medical_context += f"User Text Input: {text_input.strip()}\n"


    # -------------------------
    # REPORT INPUT (FIXED)
    # -------------------------
    if report_input is not None:

        report_path = save_report(report_input)

        report_text = analyze_report(report_path)

        medical_context += f"""
Medical Report Text:
{report_text}

Please explain what is present in this report and then provide:
1. Report Summary
2. Possible Health Concerns
3. Precautions
4. Home Remedies
5. Doctor to Consult
6. When to seek immediate medical help
"""


    # -------------------------
    # NO INPUT CASE
    # -------------------------
    if not medical_context.strip():

        return "⚠️ Please provide input using Image, Voice, Text, or Report.", None


    # -------------------------
    # AI RESPONSE
    # -------------------------
    response_text = analyze_text_query(medical_context)


    # -------------------------
    # AUDIO RESPONSE
    # -------------------------
    audio_path = text_to_speech(response_text)

    return response_text, audio_path


# -------------------------
# CLEAR BUTTON
# -------------------------
def clear_all():

    return "", None, None, None, "", None


# -------------------------
# GRADIO UI
# -------------------------
with gr.Blocks(title="Virtual Healthcare Consultation and Support System") as demo:

    gr.Markdown("# 🏥 Virtual Healthcare Consultation and Support System")
    gr.Markdown("Upload image, voice, text symptoms, or medical report to get AI health guidance.")

    with gr.Row():

        text_input = gr.Textbox(
            label="Enter Symptoms (Optional)",
            placeholder="Example: fever, cough, headache..."
        )

        audio_input = gr.Audio(
            label="Record Voice (Optional)",
            type="filepath"
        )

        image_input = gr.Image(
            label="Upload Medical Image",
            type="numpy"
        )

        report_input = gr.File(
            label="Upload Medical Report (PDF/Image)",
            file_types=[".pdf", ".png", ".jpg", ".jpeg"]
        )


    with gr.Row():

        submit_btn = gr.Button("Submit", variant="primary")

        clear_btn = gr.Button("Clear")


    response_output = gr.Textbox(
        label="Doctor's Response"
    )

    audio_output = gr.Audio(
        label="Audio Response"
    )


    # -------------------------
    # SUBMIT BUTTON
    # -------------------------
    submit_btn.click(
        fn=process_inputs,
        inputs=[text_input, audio_input, image_input, report_input],
        outputs=[response_output, audio_output]
    )


    # -------------------------
    # CLEAR BUTTON
    # -------------------------
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[text_input, audio_input, image_input, report_input, response_output, audio_output]
    )


# -------------------------
# RUN APP
# -------------------------
if __name__ == "__main__":

    demo.launch()