from datasets import load_dataset

def load_emotion_dataset(tokenizer):
    dataset = load_dataset("emotion")
    def tokenize(example):
        return tokenizer(example["text"], padding="max_length", truncation=True)
    return dataset.map(tokenize, batched=True)
