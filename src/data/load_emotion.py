from datasets import load_dataset

def load_emotion_dataset(tokenizer, max_length=None):
    dataset = load_dataset("emotion")
    def tokenize(example):
        ml = max_length if max_length is not None else tokenizer.model_max_length
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=ml)
    dataset = dataset.map(tokenize, batched=True)
    # Take a small subset for quick runs
    #dataset["train"] = dataset["train"].select(range(500))
    #dataset["validation"] = dataset["validation"].select(range(100))
    return dataset