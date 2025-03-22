from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from langchain_community.vectorstores import Chroma
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import Optional, Literal
from src.get_data import process_and_store_text_data
from src.upload_files import create_vectorstore, sanitize_filename, delete_pdf_from_retriever, delete_pdf_file, get_pdfs_by_user, is_pdf_owned_by_user
from src.chat import *
from src.lora.fine_tuning import *
from src.lora.model_utils import *
from src.lora.ollama_utils import *
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import uvicorn
from langchain.memory import ConversationBufferMemory
import os
from fastapi.security import OAuth2PasswordBearer,OAuth2PasswordRequestForm
from config.auth import *
from config.database import *
from unsloth import FastLanguageModel

app = FastAPI()

# Crear la carpeta "users" si no existe
os.makedirs("users", exist_ok=True)

# Configuración inicial del modelo
max_seq_length = 20000
dtype = None
load_in_4bit = True

# Cargar el modelo y tokenizer inicial
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

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

class ChatMessage(BaseModel):
    msg: str  # El mensaje del usuario
    chat_id: Optional[str] = None  # ID del chat (opcional)

# Modelo para eliminar un PDF
class DeletePDFRequest(BaseModel):
    filename: str


class PDFFineTuningRequest(BaseModel):
    file: UploadFile = File(...)

class HistoricalFineTuningRequest(BaseModel):
    file: UploadFile = File(...)

# Endpoint para fine-tuning con datos de PDF
@app.post("/fine-tune-pdf/", tags=["Fine-tuning"])
async def fine_tune_pdf_endpoint(file: UploadFile = File(...)):
    try:
        # Guardar el archivo temporalmente
        file_location = f"./temp/{file.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Llamar a la función de fine-tuning
        result = fine_tune_pdf(file_location)
        return JSONResponse(content={"message": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Eliminar el archivo temporal
        if os.path.exists(file_location):
            os.remove(file_location)

# Endpoint para fine-tuning con datos históricos
@app.post("/fine-tune-historical/", tags=["Fine-tuning"])
async def fine_tune_historical_endpoint(file: UploadFile = File(...)):
    try:
        # Guardar el archivo temporalmente
        file_location = f"./temp/{file.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Llamar a la función de fine-tuning
        result = fine_tune_historical(file_location)
        return JSONResponse(content={"message": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Eliminar el archivo temporal
        if os.path.exists(file_location):
            os.remove(file_location)

# Endpoint para subir el modelo a Ollama
@app.post("/upload-to-ollama/" , tags=["Fine-tuning"])
async def upload_to_ollama_endpoint():
    try:
        # Cargar el modelo más reciente si existe
        loaded_model, loaded_tokenizer = load_latest_model()
        if loaded_model is not None and loaded_tokenizer is not None:
            global model, tokenizer
            model, tokenizer = loaded_model, loaded_tokenizer

        # Subir el modelo a Ollama
        result = upload_to_ollama(model, tokenizer)
        return JSONResponse(content={"message": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    # Si no hay un chat_id en la solicitud, generar uno nuevo
    if not message.chat_id:
        chat_id = generate_chat_id(current_user.username)
    else:
        chat_id = message.chat_id

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
    # Generar un nuevo chat_id
    chat_id = generate_chat_id(current_user.username)

    # Limpiar la memoria de la conversación actual
    conversation_memory.clear()

    return {"message": "Nuevo chat creado correctamente.", "chat_id": chat_id}
    
@app.get("/chat-history/{chat_id}", tags=["Chat"])
async def get_chat_history_endpoint(
    chat_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    Recupera el historial de un chat específico.
    """
    chat_history = load_chat_history(chat_id, current_user.username)
    return {"chat_id": chat_id, "history": chat_history}

@app.get("/chat-list/", tags=["Chat"])
async def get_chat_list(current_user: User = Depends(get_current_user)):
    """
    Obtiene una lista de todos los chats del usuario.

    Args:
        current_user (User): El usuario actual.

    Returns:
        list: Lista de chats con su ID, título y timestamp.
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
        dict: Mensaje de confirmación y el historial del chat.
    """
    # Cargar el historial del chat
    chat_history = load_chat_history(chat_id, current_user.username)

    # Reconstruir la memoria de la conversación
    load_existing_chat(chat_id, current_user.username, conversation_memory)

    return {
        "message": "Chat cargado correctamente.",
        "chat_id": chat_id,
        "history": chat_history
    }



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)