import panel as pn
from langchain.chains import ConversationChain
from langchain_community.llms import Ollama

pn.extension()

# Replace with your Mistral config

llm = Ollama(model="mistral")
conversation = ConversationChain(llm=llm)

chat_box = pn.widgets.TextAreaInput(placeholder="Ask me anything...", height=100)
output_area = pn.pane.Markdown("### Output:")


def respond(event):
    response = conversation.run(chat_box.value)
    output_area.object = f"### Response:\n{response}"


chat_box.param.watch(respond, "value")

pn.Column("## Chat with Mistral", chat_box, output_area).servable()
