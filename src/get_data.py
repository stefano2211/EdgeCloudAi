from fastapi import HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
import httpx



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



async def get_weather_data(location: str = "Madrid") -> dict:
    """
    Obtiene datos del clima desde la API de wttr.

    Args:
        location (str): Ubicación para la cual se obtendrá el clima. Por defecto es "Madrid".

    Returns:
        dict: Un diccionario con los datos del clima.
    """
    url = f"https://wttr.in/{location}?format=j1"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            weather_data = response.json()
            return weather_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener datos del clima: {e}")



def preprocess_weather_data(weather_data: dict) -> str:
    """
    Preprocesa los datos del clima para convertirlos en un formato de texto.

    Args:
        weather_data (dict): Datos del clima en formato JSON.

    Returns:
        str: Texto preprocesado con los datos del clima.
    """
    current_condition = weather_data["current_condition"][0]
    weather_text = (
        f"Ubicación: {weather_data['nearest_area'][0]['areaName'][0]['value']}\n"
        f"Temperatura: {current_condition['temp_C']}°C\n"
        f"Condición: {current_condition['weatherDesc'][0]['value']}\n"
        f"Humedad: {current_condition['humidity']}%\n"
        f"Viento: {current_condition['windspeedKmph']} km/h\n"
        f"Date: {current_condition['localObsDateTime']}"
    )
    return weather_text


async def process_and_store_weather_data(vectorstore, location: str = "Madrid") -> None:
    """
    Procesa y almacena datos del clima en un vectorstore.

    Args:
        vectorstore: El vectorstore donde se almacenarán los documentos.
        location (str): Ubicación para la cual se obtendrá el clima. Por defecto es "Madrid".
    """
    weather_data = await get_weather_data(location)
    if weather_data:
        weather_text = preprocess_weather_data(weather_data)
        doc = Document(weather_text, metadata={"source": "wttr.in", "location": location})

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
        split_docs = text_splitter.split_documents([doc])
        vectorstore.add_documents(split_docs)