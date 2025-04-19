import argparse
import json

from transformers import AutoTokenizer, Trainer, TrainingArguments

from data.load_imdb import load_imdb_dataset
from models.models import create_model

from sklearn.metrics import accuracy_score, f1_score 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="lora", choices=["none", "lora", "prefix", "prompt"])
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    return parser.parse_args()

def save_results(method, results, trainable_params, args, filename="results.json"):  
    entry = {
        "method": method,
        "accuracy": results.get("eval_accuracy", None),
        "f1": results.get("eval_f1", None),
        "trainable_params": trainable_params,
        "parameters": {
            "model_name": args.model_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "output_dir": args.output_dir
        }
    }
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []
    data.append(entry)
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def compute_metrics(eval_pred):  
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # For prompt or prefix tuning, reduce the max_length by the number of virtual tokens (20)
    if args.method in ["prefix", "prompt"]:
        effective_max_length = tokenizer.model_max_length - 20
    else:
        effective_max_length = tokenizer.model_max_length

    dataset = load_imdb_dataset(tokenizer, max_length=effective_max_length)

    model = create_model(args.model_name, num_labels=6, method=args.method)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        report_to="none",
        warmup_steps=100,
        lr_scheduler_type="constant"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    results = trainer.evaluate()
    print("Evaluation:", results)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    save_results(args.method, results, trainable_params, args)

if __name__ == "__main__":
    main()