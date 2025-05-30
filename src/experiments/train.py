import argparse
import json
import time
import torch
import wandb
import evaluate
import numpy as np

from transformers import AutoTokenizer, Trainer, Seq2SeqTrainer, TrainingArguments, EarlyStoppingCallback, DataCollatorForSeq2Seq,  Seq2SeqTrainingArguments, GenerationConfig
from data.load_emotion import load_emotion_dataset
from data.load_style_dataset import load_style_dataset
from models.models import create_model
from sklearn.metrics import accuracy_score, f1_score

#load metrics for seq2seq
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="prefix", choices=["none", "lora", "prefix", "prompt"])
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    return parser.parse_args()


def compute_metrics_seq2seq(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    global tokenizer
    predictions = np.where(predictions == -100, tokenizer.pad_token_id, predictions) # Keep this commented
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

    print("\n--- Detailed Generation Debug ---")
    for i in range(min(3, len(predictions))): # Print details for first 3 samples
        print(f"Sample {i+1} Raw Prediction Tokens: {predictions[i][:30]}") # Print first 30 tokens
        decoded_pred_no_skip = tokenizer.decode(predictions[i], skip_special_tokens=False)
        print(f"Sample {i+1} Decoded Prediction (no skip): '{decoded_pred_no_skip}'")
        decoded_pred_skip = tokenizer.decode(predictions[i], skip_special_tokens=True)
        print(f"Sample {i+1} Decoded Prediction (skip_special_tokens=True): '{decoded_pred_skip}'")

        processed_pred_for_bertscore = decoded_pred_skip.strip()
        print(f"Sample {i+1} Processed Prediction for BERTscore (after strip()): '{processed_pred_for_bertscore}' (is_empty: {not bool(processed_pred_for_bertscore)})")

        decoded_label_skip = tokenizer.decode(labels[i], skip_special_tokens=True) # Assuming labels are already processed for -100
        print(f"Sample {i+1} Decoded Label (skip_special_tokens=True): '{decoded_label_skip}'")
    print("--- End Detailed Generation Debug ---\n")

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Original print for quick overview
    print("Sample Predictions and Labels (batch_decode):")
    for i in range(min(10, len(decoded_preds))):
        print(f"Prediction {i+1}: '{decoded_preds[i]}'")
        print(f"Reference      {i+1}: '{decoded_labels[i]}'")

    processed_preds = [pred.strip() for pred in decoded_preds]
    processed_labels_for_bleu = [[label.strip()] for label in decoded_labels]
    processed_labels_for_others = [label.strip() for label in decoded_labels]

    non_empty_preds_for_bertscore = [p for p in processed_preds if p]
    print(f"Number of non-empty predictions for BERTscore: {len(non_empty_preds_for_bertscore)} out of {len(processed_preds)}")


    bleu = bleu_metric.compute(predictions=processed_preds, references=processed_labels_for_bleu)
    rouge = rouge_metric.compute(predictions=processed_preds, references=processed_labels_for_others)

    bs_preds, bs_refs = [], []
    for pred, ref in zip(processed_preds, processed_labels_for_others):
        if pred and ref:
            bs_preds.append(pred)
            bs_refs.append(ref)
    
    print(f"Number of pairs for BERTscore calculation: {len(bs_preds)}")

    bertscore = bertscore_metric.compute(predictions=bs_preds, references=bs_refs, lang="en") if bs_preds else {"precision": [0.0], "recall": [0.0], "f1": [0.0]}

    return {
        "bleu": bleu["bleu"],
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "bertscore_precision": np.mean(bertscore["precision"]),
        "bertscore_recall": np.mean(bertscore["recall"]),
        "bertscore_f1": np.mean(bertscore["f1"]),
    }

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
        group=args.method,
        config=vars(args)
    )
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"Tokenizer: {tokenizer.__class__}")
    print(f"Tokenizer pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    print(f"Tokenizer eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")
    print(f"Tokenizer bos_token: {tokenizer.bos_token}, bos_token_id: {tokenizer.bos_token_id}") # Beginning of sentence
    print(f"Tokenizer model_max_length: {tokenizer.model_max_length}")

    if args.method in ["prefix", "prompt"]:
        effective_max_length = tokenizer.model_max_length - 100
    else:
        effective_max_length = tokenizer.model_max_length

    # Check if the task is classification or seq2seq
    is_seq2seq = "t5" in args.model_name.lower() or "seq2seq" in args.model_name.lower()
    
    if is_seq2seq:
        dataset = load_style_dataset(tokenizer, max_length=effective_max_length)
        model = create_model(args.model_name, method=args.method, task_type="SEQ_2_SEQ_LM")
        print(f"Model config decoder_start_token_id: {model.config.decoder_start_token_id}")
        print(f"Model config eos_token_id: {model.config.eos_token_id}")
        print(f"Model config pad_token_id: {model.config.pad_token_id}")
        print(f"Model config max_length for generation (default): {model.config.max_length}")
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        compute_metrics_fn = compute_metrics_seq2seq
    else:
        dataset = load_emotion_dataset(tokenizer, max_length=effective_max_length)
        model = create_model(args.model_name, num_labels=6, method=args.method)
        data_collator = None
        compute_metrics_fn = compute_metrics

    if args.method in ["lora", "prefix", "prompt"]:
        model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        report_to=[],
        warmup_steps=100,
        lr_scheduler_type="constant",
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
    )

    generation_config = GenerationConfig(
        max_new_tokens=64,  
        min_length=5,      
        num_beams=4,        
        early_stopping=True, 
    )

    seq_training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        report_to=["wandb"],
        warmup_steps=100,
        lr_scheduler_type="constant",
        eval_strategy="steps",
        eval_steps=2000,
        predict_with_generate=True,
        #load_best_model_at_end=True,
        generation_config = generation_config, 
    )

    if is_seq2seq:
        trainer = Seq2SeqTrainer(
            model=model,
            args=seq_training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_fn,
            #callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            data_collator=data_collator,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            data_collator=data_collator
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