from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever



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

    # Cargar el vectorstore de textos
    vectorstore_txt = Chroma(
        embedding_function=embed_model,
        persist_directory="./db/text_db",
        collection_name="text_data"
    )

    # Crear un retriever para PDFs filtrado por usuario
    retriever_pdf = vectorstore_pdf.as_retriever(
        search_kwargs={'k': 3, 'filter': {"username": username}}  # Filtrar por usuario
    )

    # Crear un retriever para textos (compartido entre usuarios)
    retriever_txt = vectorstore_txt.as_retriever(search_kwargs={'k': 3})

    # Crear el EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_pdf, retriever_txt],
        weights=[0.5, 0.5]  # Pesos iguales para ambos retrievers
    )

    # Verificar si el usuario tiene documentos en el vectorstore de PDFs
    collection_pdf = vectorstore_pdf._client.get_collection("pdf_data")
    user_docs = collection_pdf.get(where={"username": username}, include=["metadatas", "documents"])

    if not user_docs or not user_docs.get("metadatas"):
        # Si el usuario no tiene documentos, devolver un mensaje
        return "No tienes archivos PDF cargados. Por favor, sube un archivo PDF para poder consultarlo."

    # Depuración: Imprimir los documentos recuperados
    print("Documentos recuperados para el usuario:", user_docs["documents"])

    custom_prompt_template = """
    Eres un asistente que responde preguntas basadas en los archivos PDF que el usuario ha subido.
    Solo puedes responder preguntas sobre los archivos PDF que el usuario actual ha cargado.
    Si la pregunta no está relacionada con los archivos del usuario, responde que no tienes esa información.

    Aquí está el contexto relevante de los archivos del usuario:
    {context}

    Historial de conversación:
    {history}

    Pregunta:
    {question}

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