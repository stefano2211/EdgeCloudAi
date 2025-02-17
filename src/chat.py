from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS  # Cambiamos Chroma por FAISS
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from langchain.docstore.document import Document
import os

def chat(msg: str, reset: bool = False) -> str:
    """
    Genera una respuesta a un mensaje utilizando un modelo de lenguaje y un vectorstore.
    Si reset es True, se borra el buffer de la conversación.

    Args:
        msg (str): El mensaje del usuario.
        reset (bool): Si es True, se resetea la memoria de la conversación.

    Returns:
        str: La respuesta generada por el modelo.
    """
    conversation_memory = ConversationBufferMemory(memory_key="history", input_key="question")

    # Resetear la memoria si el usuario lo solicita
    if reset:
        conversation_memory.clear()
        return "La conversación ha sido reseteada. ¿En qué puedo ayudarte?"

    # Inicializar el modelo de lenguaje
    llm = Ollama(model="llama3.1:8b")

    # Inicializar el modelo de embeddings
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Cargar o inicializar el vectorstore para PDF
    if os.path.exists("./db/pdf_db/index.faiss"):
        vectorstore_pdf = FAISS.load_local("./db/pdf_db", embed_model, allow_dangerous_deserialization=True)
    else:
        # Si no existe, inicializar con un documento ficticio
        dummy_doc_pdf = Document(
            "Documento inicial para inicializar FAISS (PDF)",
            metadata={"source": "inicialización"}
        )
        vectorstore_pdf = FAISS.from_documents([dummy_doc_pdf], embed_model)
        vectorstore_pdf.save_local("./db/pdf_db")

    # Cargar o inicializar el vectorstore para datos del clima
    if os.path.exists("./db/weather_db/index.faiss"):
        vectorstore_weather = FAISS.load_local("./db/weather_db", embed_model,allow_dangerous_deserialization=True)
    else:
        # Si no existe, inicializar con un documento ficticio
        dummy_doc_weather = Document(
            "Documento inicial para inicializar FAISS (Clima)",
            metadata={"source": "inicialización"}
        )
        vectorstore_weather = FAISS.from_documents([dummy_doc_weather], embed_model)
        vectorstore_weather.save_local("./db/weather_db")

    # Configurar los retrievers
    retriever_pdf = vectorstore_pdf.as_retriever(search_kwargs={'k': 3})
    retriever_weather = vectorstore_weather.as_retriever(search_kwargs={'k': 3})

    # Configurar el EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_pdf, retriever_weather],
        weights=[0.5, 0.5]  # Pesos iguales para ambos retrievers
    )

    # Plantilla de prompt personalizada
    custom_prompt_template = """
    Si la pregunta es sobre los datos del clima, responde basado en los datos meteorológicos.
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

    # Configurar la cadena de RetrievalQA
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=ensemble_retriever,  # Usar el EnsembleRetriever aquí
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": conversation_memory,
        }
    )

    # Generar la respuesta
    response = qa.invoke({"query": msg})
    return response['result']