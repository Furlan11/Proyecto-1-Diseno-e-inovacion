from core import run_llm
import streamlit as st
from streamlit_chat import message

st.header("LangChain - Documentation Helper Bot")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here")

# Asegurando la inicialización de los estados de sesión
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def create_sources_string(source_urls: set[str]) -> str:
    if not source_urls:
        return ""
    return "Sources:\n" + "\n".join(f"{i + 1}. {src}" for i, src in enumerate(sorted(source_urls)))

if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )

    sources = {doc.metadata["source"] for doc in generated_response["source"]}
    formatted_response = (
        f"{generated_response['result']}\n\n{create_sources_string(sources)}"
    )

    st.session_state["user_prompt_history"].append(prompt)
    st.session_state["chat_answers_history"].append(formatted_response)
    st.session_state["chat_history"].append(("human", prompt))
    st.session_state["chat_history"].append(("ai", generated_response["result"]))

if st.session_state["chat_answers_history"]:
    for user_query, response in zip(
        st.session_state["user_prompt_history"], st.session_state["chat_answers_history"]
    ):
        message(user_query, is_user=True)
        message(response)
