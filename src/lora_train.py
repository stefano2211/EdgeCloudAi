# src/lora_training.py
# src/lora_training.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch
import os
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
import torch
import os
from src.upload_file import load_json_data

# Configuración del modelo y tokenizer
MODEL_NAME = "distilbert/distilgpt2"
MODEL_DIR = "./models/lora_adjusted_model"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train_lora_incremental(json_file_path, learning_rate=5e-5, epochs=10, batch_size=4):
    """
    Ajusta el modelo con LoRA usando un archivo JSON con pares de input y output.

    Args:
        json_file_path (str): Ruta al archivo JSON con pares de input y output.
        learning_rate (float): Tasa de aprendizaje para el entrenamiento.
        epochs (int): Número de épocas de entrenamiento.
        batch_size (int): Tamaño del lote (batch size).

    Returns:
        El modelo ajustado con LoRA.
    """
    # Cargar el archivo JSON
    data = load_json_data(json_file_path)

    # Cargar el modelo y tokenizer preentrenados
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Configurar el pad_token si no está definido
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configuración de LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["attn.c_attn", "attn.c_proj"],  # Módulos específicos de distilgpt2
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Aplicar LoRA al modelo
    model = get_peft_model(model, lora_config)

    # Preparar los datos de entrenamiento
    inputs = [item["input"] for item in data]
    outputs = [item["output"] for item in data]

    # Combinar inputs y outputs para el entrenamiento
    combined_texts = [f"Input: {input}\nOutput: {output}" for input, output in zip(inputs, outputs)]

    # Crear un dataset
    dataset = Dataset.from_dict({"text": combined_texts})

    # Tokenizar el dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Dividir el dataset en entrenamiento y validación
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    # Configurar los argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",  # Evaluar al final de cada época
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Usar mixed precision si hay GPU
    )

    # Configurar el DataCollator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # No usar masked language modeling
    )

    # Crear el Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Agregar conjunto de validación
        data_collator=data_collator,
    )

    # Entrenar el modelo
    print("Iniciando el entrenamiento...")
    trainer.train()

    # Guardar el modelo ajustado
    save_lora_model(model, tokenizer)

    return model


def save_lora_model(model, tokenizer, model_dir="./models/lora_adjusted_model"):
    """
    Guarda el modelo ajustado con LoRA y su tokenizer en una ubicación fija.

    Args:
        model: El modelo ajustado con LoRA.
        tokenizer: El tokenizer asociado al modelo.
        model_dir (str): Directorio donde se guardará el modelo.
    """
    # Crear el directorio si no existe
    os.makedirs(model_dir, exist_ok=True)

    # Guardar el modelo
    model.save_pretrained(model_dir)

    # Guardar manualmente la configuración
    config = model.config.to_dict()
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f)

    # Guardar el tokenizer
    tokenizer.save_pretrained(model_dir)

def load_lora_model():
    """
    Carga el modelo ajustado con LoRA y su tokenizer desde una ubicación fija.

    Returns:
        El modelo y tokenizer ajustados.
    """
    # Cargar el modelo y el tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # Configurar el pad_token si no está definido
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Usar el token de fin de secuencia como pad_token

    return model, tokenizer