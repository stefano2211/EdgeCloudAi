import os
from unsloth import FastLanguageModel
import torch
from torch.nn import DataParallel

MAX_SEQ_LENGTH = 20000
DTYPE= None
LOAD_IN_4BIT = True

def load_latest_model():
    """
    Carga el modelo más reciente si existe, de lo contrario devuelve None.
    """
    if os.path.exists("trained_model"):
        print("Cargando modelo previamente entrenado...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="trained_model",
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
        )
        return model, tokenizer
    else:
        print("No se encontró un modelo previamente entrenado. Entrenando desde cero...")
        return None, None

def save_model(model, tokenizer):
    """
    Guarda el modelo y el tokenizer en la carpeta trained_model.
    """
    # Liberar memoria de la GPU antes de guardar
    torch.cuda.empty_cache()


    # Acceder al modelo original si está envuelto en DataParallel
    model_to_save = model.module if isinstance(model, DataParallel) else model

    # Guardar el modelo
    model_to_save.save_pretrained("trained_model")
    tokenizer.save_pretrained("trained_model")
    print("Modelo guardado en trained_model.")