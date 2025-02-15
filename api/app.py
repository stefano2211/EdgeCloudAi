from fastapi import FastAPI, File, UploadFile, HTTPException
from langchain_community.vectorstores import Chroma
from datetime import datetime
from pydantic import BaseModel
from typing import Literal
from src.get_data import process_and_store_weather_data
from src.upload_files import create_vectorstore, sanitize_filename, delete_pdf_from_retriever, delete_pdf_file
from src.chat import chat
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import uvicorn
import os


app = FastAPI()

weather_service_status = "apagado"

class WeatherControl(BaseModel):
    status: Literal["prendido", "apagado"]
    location: str = "Madrid"  # Ubicación por defecto

class ChatMessage(BaseModel):
    msg: str
    reset: bool = False  # Opción para resetear la conversación

class DeletePDFRequest(BaseModel):
    filename: str
    

@app.post("/weather/")
async def control_weather_service(control: WeatherControl):
    """
    Endpoint para controlar el servicio de obtención de datos del clima.

    Args:
        control (WeatherControl): Objeto con el estado del servicio ("prendido" o "apagado") y la ubicación.

    Returns:
        dict: Mensaje de confirmación del estado del servicio.
    """
    global weather_service_status
    weather_service_status = control.status

    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore_weather = Chroma(
        embedding_function=embed_model,
        persist_directory="./db/weather_db",
        collection_name="weather_data"
    )

    if control.status == "prendido":
        return {"message": "El servicio está prendido. Último dato ya fue almacenado."}
    else:
        await process_and_store_weather_data(vectorstore_weather, control.location)
        return {"message": "Datos del clima obtenidos y almacenados correctamente."}
    
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint para subir un archivo PDF.

    Args:
        file (UploadFile): El archivo PDF a subir.

    Returns:
        dict: Información sobre el archivo subido.
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

    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    create_vectorstore(file_location)

    return {
        "pdf_name": safe_filename,
        "Content-Type": file.content_type,
        "file_location": file_location,
        "file_size": f"{file.size / 1_048_576:.2f} MB",
        "upload_time": upload_time,
    }

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

@app.post("/delete-pdf/")
async def delete_pdf(request: DeletePDFRequest):
    """
    Endpoint para borrar un archivo PDF del retriever y del servidor.
    """
    filename = sanitize_filename(request.filename)
    
    try:
        # Borrar los embeddings del vectorstore
        delete_pdf_from_retriever(filename)
        
        # Borrar el archivo físico del servidor
        delete_pdf_file(filename)
        
        return {"message": f"El archivo '{filename}' ha sido borrado correctamente."}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al borrar el archivo: {str(e)}")


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)