from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForSeq2SeqLM
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    IA3Config,
    TaskType
)

def create_model(model_name, num_labels=None, method="none", task_type="SEQ_CLS"):
    if task_type == "SEQ_CLS":
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        peft_task_type = TaskType.SEQ_CLS
    elif task_type == "SEQ_2_SEQ_LM":
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        peft_task_type = TaskType.SEQ_2_SEQ_LM
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    if method == "lora":
        if "t5" in model_name.lower():
            target_modules = ["q", "v"]  # For T5 models
        else:
            target_modules = ["query", "value"]
            
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=target_modules,
            task_type=peft_task_type
        )
    elif method == "prefix":
        peft_config = PrefixTuningConfig(
            task_type=peft_task_type,
            num_virtual_tokens=100,
            num_layers=12,
            token_dim=768,
            num_attention_heads=12
        )
    elif method == "prompt":
        peft_config = PromptTuningConfig(
            task_type=peft_task_type,
            num_virtual_tokens=100,
            num_layers=12,
            token_dim=768,
            num_attention_heads=12
        )
    elif method == "ia3":

        if "t5" in model_name.lower():

            target_modules = ["k", "v", "wo"]
        else:
            target_modules = ["key", "value", "output.dense"]
            
        peft_config = IA3Config(
            target_modules=target_modules,
            feedforward_modules=["wo"] if "t5" in model_name.lower() else ["output.dense"],
            task_type=peft_task_type,
        )
    else:
        return base_model

    return get_peft_model(base_model, peft_config)