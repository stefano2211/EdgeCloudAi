from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from langchain_community.vectorstores import Chroma
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import Optional, Literal
from src.get_data import process_and_store_text_data
from src.upload_files import create_vectorstore, sanitize_filename, delete_pdf_from_retriever, delete_pdf_file, get_pdfs_by_user, is_pdf_owned_by_user
from src.chat import chat, save_chat_message, get_chat_history,generate_chat_id
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import uvicorn
from langchain.memory import ConversationBufferMemory
import os
from fastapi.security import OAuth2PasswordBearer,OAuth2PasswordRequestForm
from config.auth import *
from config.database import *

app = FastAPI()

# Crear la carpeta "users" si no existe
os.makedirs("users", exist_ok=True)

processed_pdfs = []

conversation_memory = ConversationBufferMemory(memory_key="history", input_key="question")

# Modelo Pydantic para el registro de usuarios
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

# Modelo Pydantic para el inicio de sesión (con los parámetros que mencionas)
class UserLogin(BaseModel):
    grant_type: str = "password"
    username: str
    password: str

class TokenRequest(BaseModel):
    token: str

# Modelo para el control de texto
class TextControl(BaseModel):
    text: str  # Texto que se va a procesar
    metadata: dict = None  # Metadatos opcionales

# Modelo para el mensaje de chat
class ChatMessage(BaseModel):
    msg: str  # Solo el mensaje del usuario

# Modelo para eliminar un PDF
class DeletePDFRequest(BaseModel):
    filename: str

# Endpoint para registrar un nuevo usuario
@app.post("/register/", tags=["Autentication"])
async def register(user: UserCreate):
    db = next(get_db())
    db_user = get_user(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {"message": "User registered successfully"}

@app.post("/logout", tags=["Autentication"])
async def logout():
    return JSONResponse(
        content={"message": "Logged out successfully"},
        status_code=status.HTTP_200_OK,
    )

# Endpoint para iniciar sesión y obtener un token de acceso
@app.post("/token", tags=["Autentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    db = next(get_db())
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", tags=["Autentication"])
async def read_users_me(
    current_user: User = Depends(get_current_active_user),
):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "disabled": current_user.disabled,
    }


# Endpoint para procesar y almacenar texto
@app.post("/text/", tags=["External MES Data"])
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
@app.post("/upload/", tags=["Files"])
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="¡Por favor, sube un archivo PDF!")

    safe_filename = os.path.basename(file.filename)
    file_location = f"./data/{safe_filename}"

    os.makedirs(os.path.dirname(file_location), exist_ok=True)

    try:
        # Guardar el archivo PDF temporalmente
        with open(file_location, "wb") as f:
            content = await file.read()
            f.write(content)

        # Vectorizar el archivo PDF y asociarlo con el usuario
        create_vectorstore(file_location, username=current_user.username)

    except Exception as e:
        # Eliminar el archivo temporal en caso de error
        if os.path.exists(file_location):
            os.remove(file_location)
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo PDF: {str(e)}")

    finally:
        # Eliminar el archivo PDF del disco
        if os.path.exists(file_location):
            os.remove(file_location)

    # Agregar el nombre del archivo a la lista de PDFs procesados
    processed_pdfs.append(safe_filename)

    return {
        "pdf_name": safe_filename,
        "Content-Type": file.content_type,
        "file_size": f"{file.size / 1_048_576:.2f} MB",
        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

# Endpoint para generar una respuesta a un mensaje
@app.post("/chat/", tags=["Chat"])
async def chat_endpoint(
    message: ChatMessage,
    current_user: User = Depends(get_current_user),
):
    """
    Procesa un mensaje de chat y genera una respuesta.

    Args:
        message (ChatMessage): El mensaje del usuario.
        current_user (User): El usuario actual.

    Returns:
        dict: La respuesta del bot, el chat_id y el historial del chat.
    """
    # Generar un chat_id automáticamente
    chat_id = generate_chat_id(current_user.username)

    # Guardar el mensaje del usuario
    save_chat_message(chat_id, current_user.username, message.msg, role="user")

    # Generar la respuesta del bot
    bot_response = chat(msg=message.msg, buffer=conversation_memory, username=current_user.username)

    # Guardar la respuesta del bot
    save_chat_message(chat_id, current_user.username, bot_response, role="assistant")

    # Recuperar el historial del chat
    chat_history = get_chat_history(chat_id, current_user.username)

    return {
        "response": bot_response,
        "chat_id": chat_id,
        "history": chat_history
    }

# Endpoint para eliminar un PDF
@app.delete("/delete-pdf/",tags=["Files"])
async def delete_pdf(
    request: DeletePDFRequest,
    current_user: User = Depends(get_current_user),
):
    filename = sanitize_filename(request.filename)

    # Obtener la lista de PDFs del usuario actual
    user_pdfs = get_pdfs_by_user(current_user.username)

    # Verificar si el archivo existe en la lista de PDFs del usuario
    if filename not in user_pdfs:
        raise HTTPException(
            status_code=404,
            detail=f"El archivo '{filename}' no existe o no pertenece al usuario."
        )

    # Verificar que el PDF pertenece al usuario actual (opcional, redundante)
    if not is_pdf_owned_by_user(filename, current_user.username):
        raise HTTPException(
            status_code=403,
            detail="No tienes permiso para eliminar este archivo."
        )

    # Eliminar los embeddings asociados con el PDF y el usuario
    delete_pdf_from_retriever(filename, current_user.username)

    return {"message": f"El archivo '{filename}' ha sido borrado correctamente."}

# Endpoint para obtener la lista de PDFs procesados
@app.get("/get-pdfs/",tags=["Files"])
async def get_pdfs(current_user: User = Depends(get_current_user)):
    # Filtrar los PDFs que pertenecen al usuario actual
    user_pdfs = get_pdfs_by_user(current_user.username)
    return {"pdfs": user_pdfs}


@app.post("/new-chat/", tags=["Chat"])
async def new_chat(current_user: User = Depends(get_current_user)):
    """
    Crea un nuevo chat, almacena el chat anterior en Chroma y limpia el buffer.
    """
    try:
        # Obtener el historial actual del buffer
        chat_history = conversation_memory.load_memory_variables({})["history"]

        # Verificar si el historial es una lista de diccionarios
        if isinstance(chat_history, str):
            # Si es una cadena, convertirla en una lista de un solo mensaje
            chat_history = [{"text": chat_history, "role": "user"}]
        elif not isinstance(chat_history, list):
            # Si no es una lista, inicializar como una lista vacía
            chat_history = []

        # Si hay historial, guardarlo en Chroma
        if chat_history:
            chat_id = f"chat_{current_user.username}_{datetime.now().timestamp()}"
            for message in chat_history:
                # Asegurarse de que el mensaje tenga el formato correcto
                if isinstance(message, dict) and "text" in message and "role" in message:
                    save_chat_message(chat_id, current_user.username, message["text"], message["role"])
                else:
                    # Si el mensaje no tiene el formato correcto, ignorarlo o manejarlo
                    print(f"Formato de mensaje no válido: {message}")

        # Limpiar el buffer
        conversation_memory.clear()

        return {"message": "Nuevo chat creado correctamente.", "chat_id": chat_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al crear un nuevo chat: {str(e)}")
    
@app.get("/chat-history/{chat_id}", tags=["Chat"])
async def get_chat_history_endpoint(
    chat_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    Recupera el historial de un chat específico.

    Args:
        chat_id (str): El ID único del chat.
        current_user (User): El usuario actual.

    Returns:
        dict: El historial del chat.
    """
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    chat_history_collection = Chroma(
        embedding_function=embed_model,
        persist_directory="./db/chat_history_db",
        collection_name="chat_history"
    )
    # Filtrar mensajes por chat_id y username
    filter = {
        "$and": [
            {"chat_id": chat_id},
            {"username": current_user.username}
        ]
    }

    # Obtener los mensajes de Chroma
    results = chat_history_collection.get(where=filter, include=["metadatas", "documents"])

    # Formatear los resultados
    chat_history = []
    if "metadatas" in results and "documents" in results:
        for metadata, text in zip(results["metadatas"], results["documents"]):
            chat_history.append({
                "text": text,
                "role": metadata["role"],
                "timestamp": metadata["timestamp"]
            })

    return {
        "chat_id": chat_id,
        "history": chat_history
    }

@app.get("/chat-list/", tags=["Chat"])
async def get_chat_list(current_user: User = Depends(get_current_user)):
    """
    Obtiene una lista de todos los chats del usuario.

    Args:
        current_user (User): El usuario actual.

    Returns:
        list: Lista de chats con su ID y título.
    """
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    chat_history_collection = Chroma(
        embedding_function=embed_model,
        persist_directory="./db/chat_history_db",
        collection_name="chat_history"
    )
    # Filtrar mensajes por username
    filter = {"username": current_user.username}

    # Obtener los mensajes de Chroma
    results = chat_history_collection.get(where=filter, include=["metadatas", "documents"])

    # Agrupar mensajes por chat_id
    chats = {}
    if "metadatas" in results and "documents" in results:
        for metadata, text in zip(results["metadatas"], results["documents"]):
            chat_id = metadata["chat_id"]
            if chat_id not in chats:
                chats[chat_id] = {
                    "chat_id": chat_id,
                    "title": text[:50] + "..." if len(text) > 50 else text,  # Título = primeros 50 caracteres
                    "timestamp": metadata["timestamp"]
                }

    # Convertir a lista
    chat_list = list(chats.values())

    return {"chats": chat_list}

@app.post("/load-chat/", tags=["Chat"])
async def load_chat(
    chat_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    Carga un chat pasado para continuar la conversación.

    Args:
        chat_id (str): El ID único del chat.
        current_user (User): El usuario actual.

    Returns:
        dict: El historial del chat cargado.
    """
    # Recuperar el historial del chat
    chat_history = get_chat_history(chat_id, current_user.username)

    if not chat_history:
        raise HTTPException(status_code=404, detail="No se encontró el chat especificado.")

    # Limpiar el buffer de la conversación actual
    conversation_memory.clear()

    # Cargar el historial en la memoria de la conversación
    for message in chat_history:
        if message["role"] == "user":
            conversation_memory.chat_memory.add_user_message(message["text"])
        elif message["role"] == "assistant":
            conversation_memory.chat_memory.add_ai_message(message["text"])

    return {
        "message": "Chat cargado correctamente.",
        "chat_id": chat_id,
        "history": chat_history
    }




if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)