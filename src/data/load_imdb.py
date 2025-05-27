from datasets import load_dataset

def load_imdb_dataset(tokenizer, max_length):
    dataset = load_dataset("imdb")
    tokenized_dataset = dataset.map(lambda examples: tokenizer(examples["text"], truncation=True, max_length=max_length), batched=True)
    # Split the training data into training and validation sets
    split = tokenized_dataset["train"].train_test_split(test_size=0.1)  # e.g., 10% for validation
    tokenized_dataset["train"] = split["train"]
    tokenized_dataset["validation"] = split["test"]
    return tokenized_dataset
