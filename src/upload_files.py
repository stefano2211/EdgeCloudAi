from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from fastapi import HTTPException
import re
import os
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings



def process_file(filepath: str, username: str) -> tuple[list[str], list[dict]]:
    """
    Procesa un archivo PDF y devuelve los textos y metadatos extraídos.

    Args:
        filepath (str): La ruta del archivo PDF a procesar.
        username (str): El nombre de usuario que subió el archivo.

    Returns:
        tuple[list[str], list[dict]]: Lista de textos y lista de metadatos.
    """
    # Cargar el archivo PDF
    loader = PyPDFLoader(file_path=filepath)
    documents = loader.load()

    # Dividir el texto en fragmentos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    docs = text_splitter.split_documents(documents)

    # Extraer los textos y los metadatos
    texts = [doc.page_content for doc in docs]
    metadatas = [{"username": username, "filename": os.path.basename(filepath)} for _ in docs]

    return texts, metadatas

def create_vectorstore(filepath: str, username: str):
    """
    Crea un vectorstore en Chroma a partir de un archivo PDF.

    Args:
        filepath (str): La ruta del archivo PDF.
        username (str): El nombre de usuario que subió el archivo.
    """
    # Cargar el modelo de embeddings
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Inicializar o cargar el vectorstore de Chroma
    vectorstore = Chroma(
        embedding_function=embed_model,
        persist_directory="./db/pdf_db",
        collection_name="pdf_data"
    )

    # Procesar el archivo PDF
    texts, metadatas = process_file(filepath, username)

    # Verificar que se hayan extraído textos
    if not texts:
        raise ValueError("No se pudo extraer texto del archivo PDF.")

    # Agregar los textos y metadatos al vectorstore
    vectorstore.add_texts(texts, metadatas=metadatas)

def is_pdf_owned_by_user(filename: str, username: str) -> bool:
    """
    Verifica si un archivo PDF pertenece a un usuario.

    Args:
        filename (str): El nombre del archivo PDF.
        username (str): El nombre de usuario.

    Returns:
        bool: True si el archivo pertenece al usuario, False en caso contrario.
    """
    # Cargar el vectorstore de Chroma
    vectorstore = Chroma(
        embedding_function=FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        persist_directory="./db/pdf_db",
        collection_name="pdf_data"
    )

    # Obtener la colección de Chroma
    collection = vectorstore._client.get_collection("pdf_data")

    # Construir el filtro con el operador $and
    filter = {
        "$and": [
            {"filename": filename},
            {"username": username}
        ]
    }

    # Buscar el documento por nombre de archivo y usuario
    results = collection.get(where=filter, include=["metadatas"])
    if "metadatas" in results and results["metadatas"]:
        return True  # El archivo pertenece al usuario
    return False

def get_pdfs_by_user(username: str) -> list:
    """
    Obtiene los nombres de los archivos PDF que pertenecen a un usuario.

    Args:
        username (str): El nombre de usuario.

    Returns:
        list: Lista de nombres de archivos PDF únicos.
    """
    # Cargar el vectorstore de Chroma
    vectorstore = Chroma(
        embedding_function=FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        persist_directory="./db/pdf_db",
        collection_name="pdf_data"
    )

    # Obtener la colección de Chroma
    collection = vectorstore._client.get_collection("pdf_data")

    # Obtener todos los documentos con sus metadatos
    results = collection.get(include=["metadatas"])
    user_pdfs = []

    # Verificar si hay metadatos en los resultados
    if "metadatas" in results:
        for metadata in results["metadatas"]:
            if metadata and metadata.get("username") == username:
                filename = metadata.get("filename")
                if filename not in user_pdfs:  # Evitar duplicados
                    user_pdfs.append(filename)

    return user_pdfs


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

def delete_pdf_from_retriever(filename: str, username: str):
    """
    Elimina los embeddings asociados con un archivo PDF y un usuario específico.

    Args:
        filename (str): El nombre del archivo PDF.
        username (str): El nombre de usuario.
    """
    try:
        # Cargar el vectorstore de Chroma
        vectorstore = Chroma(
            embedding_function=FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            persist_directory="./db/pdf_db",
            collection_name="pdf_data"
        )

        # Obtener la colección de Chroma
        collection = vectorstore._client.get_collection("pdf_data")

        # Construir el filtro con el operador $and
        filter = {
            "$and": [
                {"filename": filename},
                {"username": username}
            ]
        }

        # Verificar si el archivo existe en Chroma
        results = collection.get(where=filter, include=["metadatas"])
        if not results.get("ids", []):
            raise HTTPException(status_code=404, detail=f"No se encontró el archivo '{filename}' en Chroma.")

        # Eliminar los documentos asociados con el archivo y el usuario
        collection.delete(where=filter)



        return {
            "message": f"El archivo '{filename}' y sus embeddings asociados han sido eliminados correctamente.",
            
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al eliminar el archivo: {str(e)}")
