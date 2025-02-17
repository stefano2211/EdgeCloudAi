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


def preprocess_weather_data(weather_data: dict, event_description: str) -> str:
    """
    Preprocesa los datos del clima y los combina con la descripción del evento.

    Args:
        weather_data (dict): Datos del clima en formato JSON.
        event_description (str): Descripción del evento proporcionada por el usuario.

    Returns:
        str: Texto preprocesado con los datos del clima y la descripción del evento.
    """
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
    """
    Procesa y almacena un evento en un vectorstore.

    Args:
        vectorstore: El vectorstore donde se almacenarán los documentos.
        event_description (str): Descripción del evento proporcionada por el usuario.
        weather_data (dict): Datos del clima en formato JSON.
        location (str): Ubicación del evento.
    """
    # Combinar los datos del clima con la descripción del evento
    event_text = preprocess_weather_data(weather_data, event_description)

    # Crear el documento con el texto contextualizado y metadatos
    doc = Document(
        event_text,
        metadata={
            "source": "API del clima",
            "event_description": event_description,
            "location": location
        }
    )

    # Dividir el documento en chunks (si es necesario)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    split_docs = text_splitter.split_documents([doc])

    # Almacenar en el vectorstore
    vectorstore.add_documents(split_docs)