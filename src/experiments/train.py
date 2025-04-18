import argparse
from transformers import AutoTokenizer, Trainer, TrainingArguments

from data.loader import load_emotion_dataset
from models.models import create_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="none", choices=["none", "lora", "prefix", "prompt"])
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./output")
    return parser.parse_args()

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = load_emotion_dataset(tokenizer)

    model = create_model(args.model_name, num_labels=6, method=args.method)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer
    )

    trainer.train()
    results = trainer.evaluate()
    print("Evaluation:", results)

if __name__ == "__main__":
    main()
