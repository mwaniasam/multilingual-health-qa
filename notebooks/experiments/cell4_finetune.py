"""
=============================================================================
CELL 4: FINE-TUNE QWEN2.5-7B-INSTRUCT WITH QLoRA (~2-3 hours on A100)
=============================================================================
Open-source, reproducible, saves to Drive.
=============================================================================
"""
import torch, gc, json
from pathlib import Path
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

FT_MODEL_DIR = OUTPUT_DIR / 'qwen-ft-health'
FT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Check if already fine-tuned (resume)
if (FT_MODEL_DIR / 'adapter_config.json').exists():
    log(f"Fine-tuned model already exists at {FT_MODEL_DIR}")
    log("Skip this cell or delete the directory to retrain.")
else:
    log("Starting fine-tuning...")
    log(f"GPU: {torch.cuda.get_device_name(0)} | "
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load training data
    train_data_path = OUTPUT_DIR / 'ft_train_data.json'
    train_texts = json.load(open(train_data_path))
    log(f"Training samples: {len(train_texts)}")

    # ---- Load model in 4-bit ----
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    try:
        from trl import SFTConfig
        HAS_SFT_CONFIG = True
    except ImportError:
        HAS_SFT_CONFIG = False
    from transformers import TrainingArguments
    from datasets import Dataset

    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    log(f"Loading {MODEL_NAME} in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    log(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    # ---- LoRA config ----
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"LoRA: {trainable/1e6:.1f}M trainable / {total/1e6:.0f}M total "
        f"({100*trainable/total:.2f}%)")

    # ---- Dataset ----
    dataset = Dataset.from_dict({"text": train_texts})
    log(f"Dataset: {len(dataset)} samples")

    # ---- Training ----
    training_kwargs = dict(
        output_dir=str(FT_MODEL_DIR / 'checkpoints'),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        optim="paged_adamw_8bit",
        group_by_length=True,
        report_to="none",
        seed=42,
    )

    if HAS_SFT_CONFIG:
        log("Using SFTConfig (TRL >= 0.8)")
        training_kwargs['max_seq_length'] = 1536
        training_kwargs['dataset_text_field'] = "text"
        training_kwargs['packing'] = False
        args = SFTConfig(**training_kwargs)
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
    else:
        log("Using TrainingArguments (TRL < 0.8)")
        args = TrainingArguments(**training_kwargs)
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            max_seq_length=1536,
            dataset_text_field="text",
            packing=False,
        )

    log(f"\nStarting training: {training_kwargs['num_train_epochs']} epochs, "
        f"batch {training_kwargs['per_device_train_batch_size']}×"
        f"{training_kwargs['gradient_accumulation_steps']} = "
        f"{training_kwargs['per_device_train_batch_size'] * training_kwargs['gradient_accumulation_steps']}")

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    log(f"Training complete in {train_time/60:.0f} minutes")

    # ---- Save ----
    log(f"Saving model to {FT_MODEL_DIR}...")
    trainer.model.save_pretrained(str(FT_MODEL_DIR))
    tokenizer.save_pretrained(str(FT_MODEL_DIR))
    log("Model saved to Drive!")

    # Cleanup
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    log("GPU memory freed.")
