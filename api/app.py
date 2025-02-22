from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from langchain_community.vectorstores import Chroma
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, Literal
from src.get_data import process_and_store_text_data
from src.upload_files import create_vectorstore, sanitize_filename, delete_pdf_from_retriever, delete_pdf_file
from src.chat import chat
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import uvicorn
from langchain.memory import ConversationBufferMemory
import os
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext
import re

app = FastAPI()

# Crear la carpeta "users" si no existe
os.makedirs("users", exist_ok=True)

# Configuración de la base de datos SQLite (guardada en la carpeta "users")
DATABASE_URL = "sqlite:///./users/test.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

processed_pdfs = []

conversation_memory = ConversationBufferMemory(memory_key="history", input_key="question")

Base = declarative_base()

# Modelo de usuario
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)

# Crear la base de datos
Base.metadata.create_all(bind=engine)

# Configuración de hashing de contraseñas
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

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
    scope: Optional[str] = ""
    client_id: Optional[str] = ""
    client_secret: Optional[str] = ""

class TokenRequest(BaseModel):
    token: str

# Configuración de JWT
SECRET_KEY = "your-secret-key"  # Debe ser una cadena segura
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Función para verificar la contraseña
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# Función para obtener el hash de la contraseña
def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

# Función para obtener un usuario por nombre de usuario
def get_user(db, username: str):
    return db.query(User).filter(User.username == username).first()

# Función para autenticar al usuario
def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

# Función para crear un token de acceso
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Función para decodificar y validar el token
def decode_token(token: str):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username

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
async def register(user: UserCreate):
    db = SessionLocal()
    db_user = get_user(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {"message": "User registered successfully"}

# Endpoint para iniciar sesión y obtener un token de acceso
@app.post("/token/")
async def login(user_login: UserLogin):
    # Validar que el grant_type sea "password"
    if user_login.grant_type != "password":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid grant_type. Only 'password' is supported.",
        )

    db = SessionLocal()
    user = authenticate_user(db, user_login.username, user_login.password)
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

# Endpoint para obtener la información del usuario actual (recibe el token en el cuerpo)
@app.post("/users/me/")
async def read_users_me(token_request: TokenRequest):
    # Decodificar y validar el token
    username = decode_token(token_request.token)

    # Obtener la información del usuario
    db = SessionLocal()
    user = get_user(db, username=username)
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