import vertexai
import streamlit as st

from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models

PROJECT_ID = "gcp-data-science-demo"
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)

model = GenerativeModel(
    "gemini-1.5-pro-001",
    system_instruction=["""You are a championship-winning NBA coach"""]
 )

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

st.title("Chat with Gemini")

if "chat" not in st.session_state:
    st.session_state["chat"] = model.start_chat()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = st.session_state["chat"].send_message(
          prompt,
          generation_config=generation_config,
          safety_settings=safety_settings
        )
        st.markdown(response.text)
    st.session_state.messages.append({"role": "assistant", "content": response.text})
