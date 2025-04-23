import argparse
import json
import time
import torch
import wandb

from transformers import AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from data.load_emotion import load_emotion_dataset
from models.models import create_model
from sklearn.metrics import accuracy_score, f1_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="lora", choices=["none", "lora", "prefix", "prompt"])
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    return parser.parse_args()

def save_results(method, results, trainable_params, args, train_time=None, max_memory=None, filename="results.json"):
    entry = {
        "method": method,
        "accuracy": results.get("eval_accuracy", None),
        "f1": results.get("eval_f1", None),
        "trainable_params": trainable_params,
        "train_time": train_time,
        "max_memory_MB": max_memory,
        "parameters": {
            "model_name": args.model_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "output_dir": args.output_dir
        }
    }
    # Save to JSON
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []
    data.append(entry)
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    # Log to wandb
    wandb.log(entry)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

def main():
    args = parse_args()
    wandb.init(
        project="peft_comparison",
        group=args.method,  # Group runs by method
        config={
            "method": args.method,
            "model_name": args.model_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "output_dir": args.output_dir
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.method in ["prefix", "prompt"]:
        effective_max_length = tokenizer.model_max_length - 100
    else:
        effective_max_length = tokenizer.model_max_length

    dataset = load_emotion_dataset(tokenizer, max_length=effective_max_length)
    model = create_model(args.model_name, num_labels=6, method=args.method)
    if args.method in ["lora", "prefix", "prompt"]:
        model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        report_to="wandb",
        warmup_steps=100,
        lr_scheduler_type="constant",
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time

    results = trainer.evaluate()
    print("Evaluation:", results)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    max_memory = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None

    save_results(args.method, results, trainable_params, args, train_time=train_time, max_memory=max_memory)

if __name__ == "__main__":
    main()