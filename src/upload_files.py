from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from fastapi import HTTPException
import re
import os
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings



def process_file(filepath: str) -> list:
    """
    Procesa un archivo PDF y devuelve los documentos extraídos.
    """
    loader = PyPDFLoader(file_path=filepath)
    data_csv = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    docs = text_splitter.split_documents(data_csv)

    # Agregar el campo 'source' a los metadatos de cada documento
    filename = os.path.basename(filepath)
    for doc in docs:
        doc.metadata["source"] = filename

    return docs

def create_vectorstore(filepath: str):
    """
    Crea un vectorstore a partir de un archivo PDF.

    Args:
        filepath (str): La ruta del archivo PDF a procesar.

    Returns:
        vectorstore: El vectorstore creado a partir de los documentos extraídos.
    """
    docs = process_file(filepath)
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Verificar si ya existe un índice FAISS
    if os.path.exists("./db/pdf_db/index.faiss"):
        vectorstore_files = FAISS.load_local("./db/pdf_db", embed_model, allow_dangerous_deserialization=True)
        vectorstore_files.add_documents(docs)  # Agregar nuevos documentos al índice existente
    else:
        # Si no existe, crear un nuevo índice FAISS
        vectorstore_files = FAISS.from_documents(docs, embed_model)

    # Guardar el índice en disco
    vectorstore_files.save_local("./db/pdf_db")
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
        # Inicializar el modelo de embeddings
        embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Cargar el índice FAISS existente
        if os.path.exists("./db/pdf_db/index.faiss"):
            vectorstore_files = FAISS.load_local("./db/pdf_db", embed_model, allow_dangerous_deserialization=True)
        else:
            raise HTTPException(status_code=404, detail="No se encontró el índice FAISS.")

        # Obtener todos los documentos del índice FAISS
        docs = vectorstore_files.docstore._dict

        # Filtrar los documentos asociados al archivo
        docs_to_delete = [
            doc_id for doc_id, metadata in docs.items()
            if metadata.metadata.get("source") == filename
        ]

        # Verificar si se encontraron documentos para eliminar
        if not docs_to_delete:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontraron documentos asociados a '{filename}'. Metadatos: {[m.metadata for m in docs.values()]}"
            )

        # Eliminar los documentos del índice
        vectorstore_files.delete(docs_to_delete)

        # Guardar los cambios en el índice FAISS
        vectorstore_files.save_local("./db/pdf_db")

        return {"message": f"Se eliminaron {len(docs_to_delete)} documentos asociados a '{filename}'."}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al borrar los embeddings: {str(e)}")