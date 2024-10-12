import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeLangChain

# Cargar variables de entorno
load_dotenv()

# Inicializar la conexión con Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Any:
    # Crear embeddings y conectar con la base de datos de Pinecone
    embeddings = OpenAIEmbeddings()

    # Aquí corregimos la llamada a `from_existing_index` pasando el embedding como argumento.
    docsearch = PineconeLangChain.from_existing_index(
        index_name="langchain-doc-index", embedding=embeddings
    )

    # Inicializamos el modelo de lenguaje con ChatOpenAI
    chat = ChatOpenAI(verbose=True, temperature=0)

    # Descargar prompts predefinidos de LangChain Hub
    retrieval_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_prompt)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    # Crear un recuperador que tiene en cuenta el historial del chat
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )

    # Crear la cadena de recuperación de preguntas y respuestas
    qa = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=stuff_documents_chain
    )

    # Ejecutar la consulta
    try:
        result = qa.invoke(input={"input": query, "chat_history": chat_history})
        new_result = {
            "query": result["input"],
            "result": result["answer"],
            "source": result["context"],
        }
        return new_result
    except Exception as e:
        return {"result": f"Error: {str(e)}", "source": []}

if __name__ == "__main__":
    print(run_llm(query="What is a Chain in LangChain?"))
