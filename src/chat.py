from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from datetime import datetime
import uuid


def chat(msg: str, buffer, username: str) -> str:
    """
    Genera una respuesta a un mensaje utilizando un modelo de lenguaje y un vectorstore.

    Args:
        msg (str): El mensaje del usuario.
        buffer: Memoria de la conversación.
        username (str): El nombre de usuario actual.

    Returns:
        str: La respuesta generada por el modelo.
    """
    llm = Ollama(model="llama3.1:8b")
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Cargar el vectorstore de PDFs
    vectorstore_pdf = Chroma(
        embedding_function=embed_model,
        persist_directory="./db/pdf_db",
        collection_name="pdf_data"
    )

    # Cargar el vectorstore de textos (datos de las máquinas)
    vectorstore_txt = Chroma(
        embedding_function=embed_model,
        persist_directory="./db/text_db",
        collection_name="text_data"
    )

    # Crear un retriever para PDFs filtrado por usuario
    retriever_pdf = vectorstore_pdf.as_retriever(
        search_kwargs={'k': 3, 'filter': {"username": username}}  # Filtrar por usuario
    )

    # Crear un retriever para textos (datos de las máquinas)
    retriever_txt = vectorstore_txt.as_retriever(search_kwargs={'k': 3})

    # Crear el EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_pdf, retriever_txt],
        weights=[0.5, 0.5]  # Pesos iguales para ambos retrievers
    )

    # Definir el prompt
    custom_prompt_template = """
    Si la pregunta es sobre los datos de las máquinas, responde basado en los datos de las máquinas.
    Si la pregunta es sobre un archivo, responde basado en el contenido del archivo PDF.
    Si la pregunta necesita que combines ambos datos, hazlo para la predicción final.
    Contexto: {context}
    Historial de conversación: {history}
    Pregunta: {question}

    Responde siempre en español.
    Respuesta:
    """

    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'history', 'question']
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=ensemble_retriever,  # Usar el EnsembleRetriever aquí
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": buffer,
        }
    )

    response = qa.invoke({"query": msg})
    return response['result']


def generate_chat_id(username: str) -> str:
    """
    Genera un ID único para el chat.

    Args:
        username (str): El nombre de usuario.

    Returns:
        str: Un ID único en formato "chat_{username}_{uuid}".
    """
    unique_id = str(uuid.uuid4())  # Genera un UUID único
    return f"chat_{username}_{unique_id}"

def save_chat_message(chat_id: str, username: str, text: str, role: str = "user"):
    """
    Guarda un mensaje en el historial de chat.

    Args:
        chat_id (str): El ID único del chat.
        username (str): El nombre de usuario.
        text (str): El contenido del mensaje.
        role (str): El rol del mensaje ("user" o "assistant").
    """

    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    chat_history_collection = Chroma(
        embedding_function=embed_model,
        persist_directory="./db/chat_history_db",
        collection_name="chat_history"
    )
    # Crear metadatos para el mensaje
    metadata = {
        "chat_id": chat_id,
        "username": username,
        "timestamp": datetime.now().isoformat(),
        "role": role
    }

    # Guardar el mensaje en Chroma
    chat_history_collection.add_texts(texts=[text], metadatas=[metadata])




def get_chat_history(chat_id: str, username: str):
    """
    Recupera el historial de un chat específico.

    Args:
        chat_id (str): El ID único del chat.
        username (str): El nombre de usuario.

    Returns:
        list: Lista de mensajes en el chat.
    """
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    chat_history_collection = Chroma(
        embedding_function=embed_model,
        persist_directory="./db/chat_history_db",
        collection_name="chat_history"
    )
    # Filtrar mensajes por chat_id y username
    filter = {
        "$and": [
            {"chat_id": chat_id},
            {"username": username}
        ]
    }

    # Obtener los mensajes de Chroma
    results = chat_history_collection.get(where=filter, include=["metadatas", "documents"])

    # Formatear los resultados
    chat_history = []
    if "metadatas" in results and "documents" in results:
        for metadata, text in zip(results["metadatas"], results["documents"]):
            chat_history.append({
                "text": text,
                "role": metadata["role"],
                "timestamp": metadata["timestamp"]
            })

    return chat_history