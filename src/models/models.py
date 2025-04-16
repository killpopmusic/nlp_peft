from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, PrefixTuningConfig, PromptTuningConfig, TaskType

def create_model(model_name, num_labels, method="none"):
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    if method == "none":
        return base_model

    if method == "lora":
        peft_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_lin", "v_lin"], task_type=TaskType.SEQ_CLS)
    elif method == "prefix":
        peft_config = PrefixTuningConfig(num_virtual_tokens=10, task_type=TaskType.SEQ_CLS)
    elif method == "prompt":
        peft_config = PromptTuningConfig(num_virtual_tokens=10, tokenizer_name=model_name, task_type=TaskType.SEQ_CLS)
    else:
        raise ValueError(f"Unknown PEFT method: {method}")

    return get_peft_model(base_model, peft_config)
