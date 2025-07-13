import os
from dotenv import load_dotenv
import panel as pn
import openai
from openai import OpenAI
import base64
import io
from PIL import Image

# Your OpenAI API key here
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

pn.extension()

# Widgets
chat_input = pn.widgets.TextInput(
    name="Your message", placeholder="Ask a question about the image"
)
submit_button = pn.widgets.Button(name="Submit", button_type="primary")
image_upload = pn.widgets.FileInput(accept="image/*")
response_pane = pn.pane.Markdown("**Response:**", sizing_mode="stretch_width")
image_pane = pn.pane.PNG(height=300)

# Global variable for storing the uploaded image
current_image = None


def encode_image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

    base64_image = encode_image_to_base64(image)


def ask_gpt_with_image(image: Image.Image, question: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"**Error querying OpenAI API:** {e}"


def on_image_upload(event):
    global current_image
    try:
        current_image = Image.open(io.BytesIO(image_upload.value))
        image_pane.object = current_image
    except Exception as e:
        response_pane.object = f"**Error loading image:** {e}"


def on_submit(event=None):
    if not current_image:
        response_pane.object = "**Please upload an image before submitting.**"
        return
    if not chat_input.value.strip():
        response_pane.object = "**Please enter a message.**"
        return
    response_pane.object = "**Processing... Please wait.**"
    response = ask_gpt_with_image(current_image, chat_input.value)
    response_pane.object = f"**Response:**\n{response}"


submit_button.on_click(on_submit)
chat_input.param.watch(lambda _: on_submit(), "value")
image_upload.param.watch(on_image_upload, "value")

layout = pn.Column(
    "# Chat with GPT-4 Vision",
    chat_input,
    submit_button,
    response_pane,
    "## Upload an image",
    image_upload,
    image_pane,
)

layout.servable()
