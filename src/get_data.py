from fastapi import HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import FastEmbedEmbeddings
from langchain.docstore.document import Document
import httpx



class Document:
    def __init__(self, content: str, metadata: dict = None):
        self.page_content = content
        self.metadata = metadata if metadata is not None else {}

async def get_weather_data(location: str = "Madrid") -> dict:
    url = f"https://wttr.in/{location}?format=j1"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            weather_data = response.json()
            return weather_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener datos del clima: {e}")

def preprocess_weather_data(weather_data: dict, event_description: str) -> str:
    current_condition = weather_data["current_condition"][0]
    weather_text = (
        f"Evento: {event_description}\n"
        f"Ubicación: {weather_data['nearest_area'][0]['areaName'][0]['value']}\n"
        f"Temperatura: {current_condition['temp_C']}°C\n"
        f"Condición: {current_condition['weatherDesc'][0]['value']}\n"
        f"Humedad: {current_condition['humidity']}%\n"
        f"Viento: {current_condition['windspeedKmph']} km/h\n"
        f"Fecha: {current_condition['localObsDateTime']}"
    )
    return weather_text

async def process_and_store_event(vectorstore, event_description: str, weather_data: dict, location: str) -> None:
    event_text = preprocess_weather_data(weather_data, event_description)
    doc = Document(
        event_text,
        metadata={
            "source": "API del clima",
            "event_description": event_description,
            "location": location
        }
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    split_docs = text_splitter.split_documents([doc])
    vectorstore.add_documents(split_docs)
    vectorstore.save_local("./db/weather_db")
