from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever



def chat(msg: str, buffer) -> str:
    """
    Genera una respuesta a un mensaje utilizando un modelo de lenguaje y un vectorstore.

    Args:
        msg (str): El mensaje del usuario.

    Returns:
        str: La respuesta generada por el modelo.
    """
    llm = Ollama(model="llama3.1:8b")
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore_pdf = Chroma(
        embedding_function=embed_model,
        persist_directory="./db/pdf_db",
        collection_name="pdf_data"
    )

    vectorstore_txt = Chroma(
        embedding_function=embed_model,
        persist_directory="./db/text_db",
        collection_name="text_data"
    )
    
    retriever_pdf = vectorstore_pdf.as_retriever(search_kwargs={'k': 3})
    retriever_txt = vectorstore_txt.as_retriever(search_kwargs={'k': 3})
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_pdf, retriever_txt],
        weights=[0.5, 0.5]  # Pesos iguales para ambos retrievers
    )

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