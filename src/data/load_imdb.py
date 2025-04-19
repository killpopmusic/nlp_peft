from datasets import load_dataset

def load_imdb_dataset(tokenizer, max_length=None):
    dataset = load_dataset("imdb")
    def tokenize(example):
        ml = max_length if max_length is not None else tokenizer.model_max_length
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=ml)
    dataset = dataset.map(tokenize, batched=True)
    dataset["train"] = dataset["train"].select(range(1000))
    dataset["test"] = dataset["test"].select(range(200))
    return dataset