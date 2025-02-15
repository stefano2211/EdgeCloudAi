from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever



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

    llm = Ollama(model="llama3.1:8b")
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore_pdf = Chroma(
        embedding_function=embed_model,
        persist_directory="./db/pdf_db",
        collection_name="pdf_data"
    )

    vectorstore_weather = Chroma(
        embedding_function=embed_model,
        persist_directory="./db/weather_db",
        collection_name="weather_data"
    )
    
    retriever_pdf = vectorstore_pdf.as_retriever(search_kwargs={'k': 3})
    retriever_weather = vectorstore_weather.as_retriever(search_kwargs={'k': 3})
    
    ensemble_retriever = EnsembleRetriever(
    retrievers=[retriever_pdf, retriever_weather],
    weights=[0.5, 0.5]  # Pesos iguales para ambos retrievers
    )

    custom_prompt_template = """
    Si la pregunta es sobre el los datos del clima, responde basado en los datos meteorológicos.
    Si la pregunta es sobre un archivo, responde basado en el contenido del archivo PDF.
    Si la pregunta necesique que combine ambos datos hazlo para la prediccion final
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
            "memory": conversation_memory,
        }
    )


    response = qa.invoke({"query": msg})
    return response['result']