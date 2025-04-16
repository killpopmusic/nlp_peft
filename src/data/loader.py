from datasets import load_dataset

def load_emotion_dataset(tokenizer, max_length=None):
    dataset = load_dataset("emotion")
    def tokenize(example):
        # if max_length not provided, use tokenizers default max length
        ml = max_length if max_length is not None else tokenizer.model_max_length
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=ml)
    return dataset.map(tokenize, batched=True)