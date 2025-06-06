from datasets import load_dataset
import pandas as pd

import contextlib, json, torch, inspect
from collections import defaultdict
import torch, numpy as np, math, time, imagecodecs

dataset = load_dataset("/opt/models/dataset/openassistant-guanaco")

df = pd.DataFrame(dataset['train'])


import os, argparse
from datetime import datetime
import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from peft import get_peft_model

from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig
import gc


parser = argparse.ArgumentParser(description="QLoRA Finetune Configuration")
parser.add_argument("--finetune_type", choices=["full", "lora", "qlora"], default="qlora", help="Type of finetuning")
parser.add_argument("--hook", action="store_true", help="Run with compression hooks")
parser.add_argument("--autocast_adapter", action="store_true", help="Set autocast_adapter_dtype=True")
parser.add_argument("--autocast_context", action="store_true", help="Set torch.amp.autocast")
parser.add_argument("--gradient_checkpointing", action="store_true", help="Run with gradient checkpointing")
args = parser.parse_args()
print(f"{args=}")

gradient_checkpointing = args.gradient_checkpointing

output_dir="./fine-tuned_mistral"
model_name = "/opt/models/hf/Mistral-7B-v0.1"

#Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id =  tokenizer.unk_token_id
tokenizer.padding_side = 'left'

compute_dtype = getattr(torch, "bfloat16")

gc.collect()
torch.cuda.empty_cache()

if args.finetune_type == "qlora":
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, # must use this to specify bfloat16 data type !!!
            quantization_config = bnb_config,
            device_map={"": 0}
    )
    model = prepare_model_for_kbit_training(model, 
            use_gradient_checkpointing=gradient_checkpointing)
else:
    model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, # must use this to specify bfloat16 data type !!!
            device_map={"": 0}
    )


if gradient_checkpointing:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False  # for mistral or LLaMA, 在大多数 HuggingFace 的 decoder-only 模型（如 Mistral、LLaMA）中，use_cache=True 会导致模型跳过中间状态保存，从而禁用 gradient checkpointing。

if args.finetune_type in ["lora", "qlora"]:
    peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.0,
            r=16,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules= ["gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config, 
                autocast_adapter_dtype=args.autocast_adapter)   # set this to keep the adapters in bfloat16

training_arguments = SFTConfig(
        output_dir=output_dir,
        evaluation_strategy="steps",
        do_eval=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=6,
        per_device_eval_batch_size=4,
        log_level="debug",
        save_steps=100,
        logging_steps=25,
        learning_rate=2e-4,
        eval_steps=10,
        # optim='adamw_8bit',   
        optim='adamw_torch',    # use 32 bit
        bf16=True, #change to fp16 if not using an Ampere GPU
        weight_decay=0.1,
        # max_steps=100,  # = overall forward steps / gradient_accumulation_steps
        max_steps=5,
        warmup_ratio=0.01,
        lr_scheduler_type="linear",
        push_to_hub=False,  # don't push to hub
        # report_to="wandb",
        dataset_text_field="text",
        max_seq_length=512,
)


# sample one input
sample_text = dataset['train'][0]['text']      # 也可以随机选一条
encoding = tokenizer(
    sample_text,
    max_length=training_arguments.max_seq_length,
    padding='max_length',          # 填充到 512
    truncation=True,
    return_tensors='pt'
)
# 把 tensors 挪到模型所在设备
input_ids = encoding['input_ids'].to(model.device)
attention_mask = encoding['attention_mask'].to(model.device)

# 标签通常直接等于 input_ids（自回归语言模型）
inputs = {
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'labels': input_ids.clone()
}


# ----------------------------------------------------------------------
# 注册 & 简单 demo
hook_ctx = contextlib.nullcontext()
bf16_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if args.autocast_context else contextlib.nullcontext()

EVAL=True
if EVAL:
    model.eval()
else:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

if __name__ == "__main__":
    
    with hook_ctx:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()    # for testing peak memory usage
        
        for i in range(1):
            t0 = time.time()
            with bf16_ctx:
                if EVAL:
                    with torch.no_grad():
                        loss = model(**inputs).loss
                else:
                    loss = model(**inputs).loss
            print(f"Round {i}: {loss=}")
            t1 = time.time()

            # print(f"leaf & grad: {counter_table[0, 0]}")
            # print(f"leaf & no-grad: {counter_table[0, 1]}")
            # print(f"non-leaf & grad: {counter_table[1, 0]}")
            # print(f"non-leaf & no-grad: {counter_table[1, 1]}")

            t2 = time.time()

            if not EVAL:
                loss.backward()         # 不需要 optimizer.step()
                torch.cuda.synchronize()

                optimizer.step()
                optimizer.zero_grad()
                print(f"Forward time: {(t1 - t0)*1000:.2f} ms")
                print(f"Backward time: {(time.time() - t2)*1000:.2f} ms")
            print(f"Elapsed time: {(time.time() - t0)*1000:.2f} ms")

            peak_memory = torch.cuda.max_memory_allocated(device=model.device) / 1024 / 1024
            print(f"Peak memory usage: {peak_memory:.2f} MB")

            import psutil, os
            process = psutil.Process(os.getpid())
            cpu_mem_peak = process.memory_info().rss / 1024 / 1024  # 单位: MB
            print(f"Peak CPU memory usage: {cpu_mem_peak:.2f} MB")

