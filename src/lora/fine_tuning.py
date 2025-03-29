import json
from datasets import Dataset
from unsloth import apply_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from src.lora.model_utils import *
from unsloth import FastLanguageModel
from torch.nn import DataParallel
from torch.cuda.amp import autocast
from unsloth import to_sharegpt
from unsloth import standardize_sharegpt

def fine_tune_pdf(file_path):
    """
    Realiza fine-tuning con datos de un PDF en formato JSON.
    """
    global model, tokenizer  # Asegúrate de que estás utilizando las variables globales

    # Cargar el modelo más reciente si existe
    loaded_model, loaded_tokenizer = load_latest_model()
    if loaded_model is not None and loaded_tokenizer is not None:
        model, tokenizer = loaded_model, loaded_tokenizer
    else:
        # Cargar el modelo base si no existe un modelo previamente entrenado
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Meta-Llama-3.1-8B",
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
        )

        # Añadir adaptadores LoRA al modelo
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # Rango de LoRA
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

    # Configurar el modelo para usar multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Usando {torch.cuda.device_count()} GPUs.")
        model = DataParallel(model)

    # Cargar el dataset desde el archivo JSON
    with open(file_path, "r") as f:
        data = json.load(f)

    # Convertir el dataset al formato requerido por to_sharegpt
    formatted_data = []
    for d in data:
        formatted_data.append({
            "instruction": d["instruction"],
            "input": d["input"],
            "output": d["output"]
        })

    # Crear el dataset
    dataset = Dataset.from_list(formatted_data)

    # Aplicar transformación a formato conversacional
    dataset = to_sharegpt(
        dataset,
        merged_prompt="Instrucción: {instruction}\nInput: {input}",
        output_column_name="output",
        conversation_extension=3,  # 1 conversación por entrada
    )

    dataset = standardize_sharegpt(dataset)

    # Aplicar el template de chat
    chat_template = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.

    ### Instruction:
    {INPUT}

    ### Response:
    {OUTPUT}"""

    dataset = apply_chat_template(
        dataset,
        tokenizer=tokenizer,
        chat_template=chat_template,

    )

    # Configurar el entrenador
    trainer = SFTTrainer(
        model=model.module if isinstance(model, DataParallel) else model,  # Acceder al modelo original
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=20,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
        ),
    )

    # Entrenar el modelo con precisión mixta
    with autocast():
        trainer.train()

    # Guardar el modelo
    save_model(model, tokenizer)

    return "Fine-tuning completado y modelo guardado."

def fine_tune_historical(file_path):
    """
    Realiza fine-tuning con datos de un PDF en formato JSON.
    """
    global model, tokenizer  # Asegúrate de que estás utilizando las variables globales

    # Cargar el modelo más reciente si existe
    loaded_model, loaded_tokenizer = load_latest_model()
    if loaded_model is not None and loaded_tokenizer is not None:
        model, tokenizer = loaded_model, loaded_tokenizer
    else:
        # Cargar el modelo base si no existe un modelo previamente entrenado
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Meta-Llama-3.1-8B",
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
        )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    # Procesar datos JSON
    with open(file_path, "r") as f:
        data = json.load(f)

    formatted_data = []
    for record in data:
        # Construir instruction con todos los datos contextuales
        instruction = (
            "Datos de producción completos:\n"
            f"- Transacción: {record['transaction_id']} (Orden: {record['work_order_id']})\n"
            f"- Equipo: {record['equipment']} | Operador: {record['operator']}\n"
            f"- Fecha/Hora: {record['timestamp']}\n"
            "Mediciones:\n"
            f"- Temperatura: {record['sensor_data']['temperature']}°C (Límite: {record['contextual_info']['compliance_rules']['temperature_limit']}°C)\n"
            f"- Presión: {record['sensor_data']['pressure']}psi (Límite: {record['contextual_info']['compliance_rules']['pressure_limit']}psi)\n"
            f"- Vibración: {record['sensor_data']['vibration']}mm/s\n"
            "Producción:\n"
            f"- Cantidad: {record['production_metrics']['quantity']} unidades\n"
            f"- Tipo: {record['production_metrics']['product_type']}\n"
            "Contexto:\n"
            f"- Notas: {record['contextual_info']['process_notes']}"
        )

        # Construir output analítico
        output = (
            "Resultado del análisis:\n"
            f"- Probabilidad de calidad: {record['target_output']['quality_probability']:.0%}\n"
            f"- Cumplimiento: {'✅' if record['target_output']['compliance_status'] else '❌'}\n"
            f"- Explicación técnica: {record['target_output']['explanation']}"
        )

        formatted_data.append({
            "instruction": instruction,
            "input": "",  # Vacío para permitir preguntas abiertas
            "output": output
        })

    # Crear dataset
    dataset = Dataset.from_list(formatted_data)

    dataset = standardize_sharegpt(dataset)

    # Aplicar el template de chat
    chat_template = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.

    ### Instruction:
    {INPUT}

    ### Response:
    {OUTPUT}"""

    dataset = apply_chat_template(
        dataset,
        tokenizer=tokenizer,
        chat_template=chat_template,
        
    )

    # Configurar el entrenador
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=3,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=15,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )

    # Entrenar el modelo
    trainer.train()

    # Guardar el modelo
    save_model(model, tokenizer)

    return "Fine-tuning completado y modelo guardado."

