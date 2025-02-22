from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from fastapi import HTTPException
import re
import os
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings



def process_file(filepath: str) -> list:
    """
    Procesa un archivo PDF y devuelve los documentos extraídos.

    Args:
        filepath (str): La ruta del archivo PDF a procesar.

    Returns:
        list: Lista de documentos extraídos del archivo.
    """
    loader = PyPDFLoader(file_path=filepath)
    data_csv = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    docs = text_splitter.split_documents(data_csv)
    return docs

def create_vectorstore(filepath: str):
    """
    Crea un vectorstore a partir de un archivo PDF.
    """
    docs = process_file(filepath)
    
    # Añadir metadatos a los documentos
    for doc in docs:
        doc.metadata["source"] = filepath  # Asegúrate de que el campo "source" esté presente
    
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore_files = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory="./db/pdf_db",
        collection_name="pdf_data"
    )
    return vectorstore_files

def sanitize_filename(filename: str) -> str:
    """
    Sanitiza el nombre del archivo para evitar inyección de rutas.
    """
    return re.sub(r"[^a-zA-Z0-9_.-]", "", filename)

def delete_pdf_file(filename: str):
    """
    Borra el archivo PDF físico del servidor.
    """
    file_location = f"./data/{filename}"
    if os.path.exists(file_location):
        os.remove(file_location)
    else:
        raise HTTPException(status_code=404, detail=f"El archivo '{filename}' no existe en el servidor.")

def delete_pdf_from_retriever(filename: str):
    """
    Borra todos los documentos asociados a un archivo PDF del vectorstore.
    """
    try:
        # Inicializar el vectorstore
        embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore_files = Chroma(
            embedding_function=embed_model,
            persist_directory="./db/pdf_db",
            collection_name="pdf_data"
        )
        
        # Acceder a la colección de Chroma
        collection = vectorstore_files._collection
        
        # Eliminar documentos basados en los metadatos
        # Asegúrate de que el campo "source" o "filename" esté en los metadatos
        collection.delete(where={"source": filename})
        
        # Verificar si los documentos fueron eliminados
        remaining_docs = collection.get(where={"source": filename})
        if remaining_docs.get("ids"):  # Si hay documentos restantes
            raise HTTPException(
                status_code=500,
                detail=f"No se pudieron eliminar todos los documentos asociados a '{filename}'."
            )
        
        # Guardar los cambios en el disco
        vectorstore_files.persist()
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al borrar los embeddings: {str(e)}")


