import json
from datasets import Dataset
from unsloth import apply_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from src.lora.model_utils import *

def fine_tune_pdf(file_path):
    """
    Realiza fine-tuning con datos de un PDF en formato JSON.
    """
    # Cargar el modelo más reciente si existe
    loaded_model, loaded_tokenizer = load_latest_model()
    if loaded_model is not None and loaded_tokenizer is not None:
        model, tokenizer = loaded_model, loaded_tokenizer
    else:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
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

    # Cargar el dataset desde el archivo JSON
    with open(file_path, "r") as f:
        data = json.load(f)

    # Convertir el dataset al formato requerido
    formatted_data = []
    for d in data:
        conversation = [
            {"role": "user", "content": d["input"]},
            {"role": "assistant", "content": d["output"]},
        ]
        formatted_data.append({"conversations": conversation})

    # Crear el dataset
    dataset = Dataset.from_list(formatted_data)

    # Aplicar el template de chat
    chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    {OUTPUT}<|eot_id|>"""

    dataset = apply_chat_template(
        dataset,
        tokenizer=tokenizer,
        chat_template=chat_template,
        default_system_message="Eres un modelo de inteligencia artificial entrenado con manuales, documentos de la empresa e historial de máquinas para análisis y predicciones. Mis datos se actualizan de forma incremental para mejorar continuamente mi precisión.",  # Optional
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

    # Entrenar el modelo
    trainer.train()

    # Guardar el modelo
    save_model(model, tokenizer)

    return "Fine-tuning completado y modelo guardado."

def fine_tune_historical(file_path):
    """
    Realiza fine-tuning con datos históricos en formato CSV.
    """
    # Cargar el modelo más reciente si existe
    loaded_model, loaded_tokenizer = load_latest_model()
    if loaded_model is not None and loaded_tokenizer is not None:
        model, tokenizer = loaded_model, loaded_tokenizer
    else:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
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

    # Cargar el dataset desde el archivo CSV
    from datasets import load_dataset
    dataset = load_dataset(
        "csv",
        data_files=file_path,
        split="train",
    )

    # Convertir el dataset al formato requerido
    from unsloth import to_sharegpt
    dataset = to_sharegpt(
        dataset,
        merged_prompt="[[El id del modelo de la maquina: {ID}.]]"\
                      "[[\nEstatus de la maquina: {Status}.]]"\
                      "[[\nTemperatura de la maquina: {Temperatura}.]]"\
                      "[[\nFecha de registro: {Fecha}.]]"\
                      "[[\nHora de registro: {Hora}.]]",
        conversation_extension=5,
        output_column_name="Output",
    )

    # Estandarizar el dataset
    from unsloth import standardize_sharegpt
    dataset = standardize_sharegpt(dataset)

    # Aplicar el template de chat
    chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    {OUTPUT}<|eot_id|>"""

    dataset = apply_chat_template(
        dataset,
        tokenizer=tokenizer,
        chat_template=chat_template,
        default_system_message="Eres un modelo de inteligencia artificial entrenado con manuales, documentos de la empresa e historial de máquinas para análisis y predicciones. Mis datos se actualizan de forma incremental para mejorar continuamente mi precisión.",  # Optional
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

