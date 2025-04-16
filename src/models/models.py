from transformers import AutoModelForSequenceClassification
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    TaskType
)

def create_model(model_name, num_labels, method="none"):
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    if method == "lora":
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_lin", "v_lin"],
            task_type=TaskType.SEQ_CLS
        )
    elif method == "prefix":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=20,
            num_layers=6,  
            token_dim=768,
            num_attention_heads=6  # added num_attention_heads
        )
    elif method == "prompt":
        peft_config = PromptTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=20,
            num_layers=6, 
            token_dim=768,
            num_attention_heads=6  # added num_attention_heads
        )
    else:
        return base_model

    return get_peft_model(base_model, peft_config)