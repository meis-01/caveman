import panel as pn
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

pn.extension()

# LLM setup
llm = ChatOllama(model="mistral")
prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant."), ("human", "{input}")]
)
chain = prompt | llm

# Widgets
chat_box = pn.widgets.TextInput(
    name="Your message", placeholder="Type here and press Enter"
)
submit_button = pn.widgets.Button(name="Submit", button_type="primary")
output_area = pn.pane.Markdown("### Response:")


# Callback function
def respond(event=None):
    message = chat_box.value.strip()
    if not message:
        return
    output_area.object = "### Thinking..."
    result = chain.invoke({"input": message})
    output_area.object = f"### Response:\n{result.content}"
    chat_box.value = ""  # clear input


# Submit button click
submit_button.on_click(respond)

# Enter key submit
chat_box.param.watch(lambda e: respond(), "value")

# Layout
pn.Column("## Chat with Mistral", chat_box, submit_button, output_area).servable()
