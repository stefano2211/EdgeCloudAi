import subprocess
from unsloth import FastLanguageModel  
import torch


from torch.nn import DataParallel


def upload_to_ollama(model, tokenizer):
    """
    Sube el modelo más reciente a Ollama.
    Si el modelo ya existe, lo elimina antes de subir el nuevo.
    """
    model_name = "unsloth_model"  # Nombre del modelo en Ollama

    # Verificar si el modelo ya existe
    try:
        # Listar modelos en Ollama
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if model_name in result.stdout:
            print(f"El modelo '{model_name}' ya existe. Eliminándolo...")
            # Eliminar el modelo existente
            subprocess.run(["ollama", "rm", model_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error al verificar o eliminar el modelo existente: {e}")
        return f"Error al verificar o eliminar el modelo existente: {e}"

    # Liberar memoria de la GPU antes de guardar
    torch.cuda.empty_cache()


    # Guardar el modelo en formato GGUF
    try:
        # Acceder al modelo original si está envuelto en DataParallel
        model_to_save = model.module if isinstance(model, DataParallel) else model
        model_to_save.save_pretrained_gguf("model", tokenizer)
    except Exception as e:
        return f"Error al guardar el modelo en formato GGUF: {e}"

    # Subir el nuevo modelo a Ollama
    try:
        subprocess.run(["ollama", "create", model_name, "-f", "Modelfile"], check=True)
        return f"Modelo '{model_name}' subido a Ollama correctamente."
    except subprocess.CalledProcessError as e:
        return f"Error al subir el modelo a Ollama: {e}"