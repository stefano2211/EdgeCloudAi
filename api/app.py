from fastapi import FastAPI, File, UploadFile, HTTPException
from langchain_community.vectorstores import FAISS
from datetime import datetime
from pydantic import BaseModel
from typing import Literal
from src.get_data import get_weather_data, process_and_store_event
from src.upload_files import create_vectorstore, sanitize_filename, delete_pdf_from_retriever, delete_pdf_file
from src.chat import chat
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.docstore.document import Document
import uvicorn
import os


app = FastAPI()

weather_service_status = "apagado"



class WeatherControl(BaseModel):
    status: Literal["prendido", "apagado"]
    location: str = "Madrid"
    event: str = "Evento de clima"  # Descripción del evento

class ChatMessage(BaseModel):
    msg: str
    reset: bool = False  # Opción para resetear la conversación

class DeletePDFRequest(BaseModel):
    filename: str
    

@app.post("/weather/")
async def control_weather_service(control: WeatherControl):
    global weather_service_status
    weather_service_status = control.status

    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Crear el directorio si no existe
    os.makedirs("./db/weather_db", exist_ok=True)

    # Verificar si el índice FAISS ya existe
    if os.path.exists("./db/weather_db/index.faiss"):
        vectorstore_weather = FAISS.load_local("./db/weather_db", embed_model,allow_dangerous_deserialization=True)
    else:
        # Inicializar con un documento ficticio
        dummy_doc = Document(
            "Documento inicial para inicializar FAISS",
            metadata={"source": "inicialización"}
        )
        vectorstore_weather = FAISS.from_documents([dummy_doc], embed_model)
        vectorstore_weather.save_local("./db/weather_db")

    if control.status == "prendido":
        weather_data = await get_weather_data(control.location)
        await process_and_store_event(
            vectorstore_weather,
            event_description=control.event,
            weather_data=weather_data,
            location=control.location
        )
        return {"message": "El servicio está prendido. Datos del clima y evento almacenados."}
    else:
        return {"message": "El servicio está apagado."}
    
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint para subir un archivo PDF.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="¡Por favor, sube un archivo PDF!")

    safe_filename = os.path.basename(file.filename)
    file_location = f"./data/{safe_filename}"

    os.makedirs(os.path.dirname(file_location), exist_ok=True)

    try:
        with open(file_location, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocurrió un error al guardar el archivo: {str(e)}")

    # Procesar el archivo y crear el índice FAISS
    print(f"Procesando archivo: {safe_filename}")
    create_vectorstore(file_location)

    # Eliminar el archivo físico
    delete_pdf_file(safe_filename)

    return {"message": f"Archivo subido correctamente. Archivo eliminado."}

@app.post("/delete-pdf/")
async def delete_pdf(request: DeletePDFRequest):
    """
    Endpoint para borrar un archivo PDF del retriever y del servidor.
    """
    filename = sanitize_filename(request.filename)
    
    try:
        # Borrar los embeddings del vectorstore
        delete_pdf_from_retriever(filename)
        
        return {"message": f"El archivo '{filename}' ha sido borrado correctamente."}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al borrar el archivo: {str(e)}")

@app.post("/chat/")
async def quick_response(message: ChatMessage):
    """
    Endpoint para generar una respuesta a un mensaje.
    Si reset es True, se borra el historial de la conversación.

    Args:
        message (ChatMessage): El mensaje del usuario.

    Returns:
        dict: La respuesta generada.
    """
    response = chat(message.msg, reset=message.reset)
    
    
    return {"response": response}




if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)