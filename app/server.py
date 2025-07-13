import panel as pn
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.llms import Ollama
import io

pn.extension()

# Load BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# Load Mistral via Ollama
llm = Ollama(model="mistral")

# UI elements
chat_input = pn.widgets.TextInput(placeholder="Ask something about the image")
submit_button = pn.widgets.Button(name="Submit", button_type="primary")
response_pane = pn.pane.Markdown("")
image_upload = pn.widgets.FileInput(accept="image/*")
image_pane = pn.pane.Image()

# Global to store current image
current_image = None


def caption_image(image: Image.Image) -> str:
    """Generate a caption using BLIP for the given PIL image."""
    inputs = processor(images=image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def on_submit(event=None):
    global current_image
    question = chat_input.value.strip()

    if not current_image:
        response_pane.object = "**Response:**\nPlease upload an image first."
        return

    if not question:
        response_pane.object = "**Response:**\nPlease enter a question about the image."
        return

    try:
        response_pane.object = f"**Thinking...**"
        # Caption the image
        caption = caption_image(current_image)

        # Build the prompt
        prompt = f"Caption: {caption}\nQuestion: {question}\nAnswer concisely based on the caption."

        # Call LLM
        response = llm.invoke(prompt)

        # Display
        response_pane.object = f"**Response:**\n{}**Response:**\n{response}"
    except Exception as e:
        response_pane.object = f"**Response:**\nError: {e}"


def on_image_upload(event):
    global current_image
    if image_upload.value:
        image_bytes = io.BytesIO(image_upload.value)
        current_image = Image.open(image_bytes).convert("RGB")
        image_pane.object = current_image
        response_pane.object = "**Response:**\nImage uploaded. Now ask a question."


# Wire up
submit_button.on_click(on_submit)
chat_input.param.watch(lambda _: on_submit(), "value")
image_upload.param.watch(on_image_upload, "value")

# Layout
app = pn.Column(
    "## Chat with BLIP + Mistral",
    chat_input,
    submit_button,
    response_pane,
    "## Upload an image",
    image_upload,
    image_pane,
)

app.servable()
