from datasets import Dataset, DatasetDict
import ast
import os
import random

def load_style_dataset(tokenizer, max_length=None, data_dir="/home/tomek/Projects/nlp_peft/src/data/gyafc_em"):
    """
    Load style transfer dataset from local files.
    
    Args:
        tokenizer: Tokenizer for processing text
        max_length: Maximum sequence length
        data_dir: Directory containing the data files
    
    Returns:
        DatasetDict with train and test splits
    """
    # Define file paths
    train_src_path = os.path.join(data_dir, "train.src")
    train_tgt_path = os.path.join(data_dir, "train.tgt")
    test_src_path = os.path.join(data_dir, "test.src")
    test_tgt_path = os.path.join(data_dir, "test.tgt")
    
    # Check if files exist
    if not os.path.exists(train_src_path) or not os.path.exists(train_tgt_path):
        raise FileNotFoundError(f"Training files not found in {data_dir}")
    if not os.path.exists(test_src_path) or not os.path.exists(test_tgt_path):
        raise FileNotFoundError(f"Test files not found in {data_dir}")
    
    # Read source files
    with open(train_src_path, 'r', encoding='utf-8') as f:
        train_src = [line.strip() for line in f.readlines()]
    
    with open(test_src_path, 'r', encoding='utf-8') as f:
        test_src = [line.strip() for line in f.readlines()]
    
    # Read and parse target files (containing lists of formal rewrites)
    with open(train_tgt_path, 'r', encoding='utf-8') as f:
        train_tgt_raw = [line.strip() for line in f.readlines()]
        train_tgt = []
        for line in train_tgt_raw:
            try:
                # The target file contains Python-formatted lists
                targets = ast.literal_eval(line)
                if targets:
                    # Randomly select one of the formal rewrites
                    train_tgt.append(random.choice(targets))
                else:
                    train_tgt.append("")
            except (SyntaxError, ValueError):
                # If parsing fails, use the raw line
                train_tgt.append(line)
    
    with open(test_tgt_path, 'r', encoding='utf-8') as f:
        test_tgt_raw = [line.strip() for line in f.readlines()]
        test_tgt = []
        for line in test_tgt_raw:
            try:
                targets = ast.literal_eval(line)
                if targets:
                    test_tgt.append(random.choice(targets))
                else:
                    test_tgt.append("")
            except (SyntaxError, ValueError):
                test_tgt.append(line)
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        "source": train_src,
        "target": train_tgt
    })
    
    test_dataset = Dataset.from_dict({
        "source": test_src,
        "target": test_tgt
    })
    
    # Combine into a DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    
    # Limit dataset size for faster experiments (matching original code)
    train_size = min(1000, len(dataset_dict["train"]))
    test_size = min(200, len(dataset_dict["test"]))
    
    dataset_dict["train"] = dataset_dict["train"].select(range(train_size))
    dataset_dict["test"] = dataset_dict["test"].select(range(test_size))
    
    # Tokenize the datasets
    def tokenize(example):
        ml = max_length if max_length is not None else tokenizer.model_max_length
        model_inputs = tokenizer(example["source"], padding="max_length", truncation=True, max_length=ml)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(example["target"], padding="max_length", truncation=True, max_length=ml)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_dataset = dataset_dict.map(tokenize, batched=True)
    
    return tokenized_dataset