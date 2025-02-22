from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from langchain_community.vectorstores import Chroma
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import Optional, Literal
from src.get_data import process_and_store_text_data
from src.upload_files import create_vectorstore, sanitize_filename, delete_pdf_from_retriever, delete_pdf_file
from src.chat import chat
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import uvicorn
from langchain.memory import ConversationBufferMemory
import os
<<<<<<< HEAD

=======
from database.database import *
from database import crud
from database.database import engine
from sqlalchemy.orm import Session
from auth.jwt_handler import create_access_token, decode_token
import re
>>>>>>> 7dde53d25ebd80f1fa3bb62a561fc616ed2e69fa

app = FastAPI()

# Crear la carpeta "users" si no existe
os.makedirs("users", exist_ok=True)

# Crear las tablas de la base de datos
from database import models
models.Base.metadata.create_all(bind=engine)

# Configuración de la memoria de conversación
conversation_memory = ConversationBufferMemory(memory_key="history", input_key="question")

# Lista de PDFs procesados
processed_pdfs = []

# Modelo Pydantic para el registro de usuarios
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

# Modelo Pydantic para el inicio de sesión
class UserLogin(BaseModel):
    grant_type: str
    username: str
    password: str
    scope: Optional[str] = ""
    client_id: Optional[str] = ""
    client_secret: Optional[str] = ""

# Modelo Pydantic para recibir el token en el cuerpo de la solicitud
class TokenRequest(BaseModel):
    token: str

# Modelo para el control de texto
class TextControl(BaseModel):
    text: str  # Texto que se va a procesar
    metadata: dict = None  # Metadatos opcionales

# Modelo para el mensaje de chat
class ChatMessage(BaseModel):
    msg: str
    reset: bool = False  # Opción para resetear la conversación

# Modelo para eliminar un PDF
class DeletePDFRequest(BaseModel):
    filename: str

# Endpoint para registrar un nuevo usuario
@app.post("/register/")
async def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    crud.create_user(db, user)
    return {"message": "User registered successfully"}

# Endpoint para iniciar sesión y obtener un token de acceso
@app.post("/token/")
async def login(user_login: UserLogin, db: Session = Depends(get_db)):
    # Validar que el grant_type sea "password"
    if user_login.grant_type != "password":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid grant_type. Only 'password' is supported.",
        )

    user = crud.authenticate_user(db, user_login.username, user_login.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Endpoint para obtener la información del usuario actual (recibe el token en el cuerpo)
@app.post("/users/me/")
async def read_users_me(token_request: TokenRequest, db: Session = Depends(get_db)):
    # Decodificar y validar el token
    username = decode_token(token_request.token)

    # Obtener la información del usuario
    user = crud.get_user(db, username=username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "is_active": user.is_active,
    }

# Endpoint para procesar y almacenar texto
@app.post("/text/")
async def text_embed_service(control: TextControl):
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        embedding_function=embed_model,
        persist_directory="./db/text_db",
        collection_name="text_data"
    )
    await process_and_store_text_data(vectorstore, control.text, control.metadata)
    return {"message": "Datos obtenidos y almacenados correctamente."}

# Endpoint para subir un archivo PDF
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
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

    # Vectorizar el archivo PDF
    create_vectorstore(file_location)

    # Eliminar el archivo PDF del disco
    os.remove(file_location)

    # Agregar el nombre del archivo a la lista de PDFs procesados
    processed_pdfs.append(safe_filename)

    return {
        "pdf_name": safe_filename,
        "Content-Type": file.content_type,
        "file_size": f"{file.size / 1_048_576:.2f} MB",
        "upload_time": upload_time,
    }

# Endpoint para generar una respuesta a un mensaje
@app.post("/chat/")
async def quick_response(message: ChatMessage):
    response = chat(msg=message.msg, buffer=conversation_memory)
    return {"response": response}

# Endpoint para eliminar un PDF
@app.delete("/delete-pdf/")
async def delete_pdf(request: DeletePDFRequest):
    filename = sanitize_filename(request.filename)
    if filename not in processed_pdfs:
        raise HTTPException(
            status_code=404,
            detail=f"El archivo '{filename}' no está en la lista de PDFs procesados."
        )
    # Eliminar los embeddings asociados con el PDF
    delete_pdf_from_retriever(filename)
    # Eliminar el nombre del archivo de la lista de PDFs procesados
    processed_pdfs.remove(filename)
    return {"message": f"El archivo '{filename}' ha sido borrado correctamente."}

# Endpoint para obtener la lista de PDFs procesados
@app.get("/get-pdfs/")
async def get_pdfs():
    return {"pdfs": processed_pdfs}

# Endpoint para resetear el buffer de la conversación
@app.put("/reset-chat/")
async def reset_chat_buffer():
    try:
        conversation_memory.clear()  # Limpiar el buffer de la conversación
        return {"message": "El buffer del chat ha sido reseteado correctamente."}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al resetear el buffer: {str(e)}"
        )

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)