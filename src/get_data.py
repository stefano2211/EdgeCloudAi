from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from typing import Literal
from langchain.embeddings import FastEmbedEmbeddings
from langchain.vectorstores import Chroma


class Document:
    def __init__(self, content: str, metadata: dict = None):
        """
        Inicializa un documento con contenido y metadatos.

        Args:
            content (str): El contenido del documento.
            metadata (dict, optional): Metadatos asociados al documento. Por defecto es None.
        """
        self.page_content = content
        self.metadata = metadata if metadata is not None else {}

async def process_and_store_text_data(vectorstore, text: str, metadata: dict = None) -> None:
    """
    Procesa y almacena un texto en un vectorstore.

    Args:
        vectorstore: El vectorstore donde se almacenarán los documentos.
        text (str): El texto que se va a procesar y almacenar.
        metadata (dict, optional): Metadatos asociados al texto. Por defecto es None.
    """
    doc = Document(text, metadata=metadata if metadata else {})

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    split_docs = text_splitter.split_documents([doc])
    vectorstore.add_documents(split_docs)