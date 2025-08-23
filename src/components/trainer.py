from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

def build_training_objects(cfg, tokenizer, model, tokenized, compute_metrics):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=cfg.paths.models_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.training.learning_rate,
        per_device_train_batch_size=cfg.training.train_batch_size,
        per_device_eval_batch_size=cfg.training.eval_batch_size,
        num_train_epochs=cfg.training.num_epochs,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        fp16=bool(cfg.training.fp16),
        logging_dir=cfg.paths.logs_dir,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=["tensorboard"],  # nice to have
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    return trainer
