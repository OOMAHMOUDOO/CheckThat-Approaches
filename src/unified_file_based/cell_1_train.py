"""
Optimized TRL-based SFT Training for CLEF 2026 CheckThat! GRM.
"""
import os
import json
import yaml
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

# --- CONFIGURATION ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

HF_TOKEN = config["credentials"]["hf_token"] or os.environ.get("HF_TOKEN")
HF_USERNAME = "YOUR_HF_USERNAME" # <-- Replace with your Hugging Face username
# ---------------------

SYSTEM_PROMPT = (
    "You are an expert fact-checking auditor specializing in numerical, statistical, and temporal claims. "
    "Your task is to rigorously evaluate whether a previous fact-checker's verdict and justification are correct, "
    "given the original claim and retrieved evidence.\n\n"
    "Verdict Definitions:\n"
    "- True: The evidence fully supports the claim.\n"
    "- False: The evidence directly contradicts the claim.\n"
    "- Conflicting: The evidence contains contradictory elements or is inconclusive.\n\n"
    "Evaluation Criteria:\n"
    "1. Evidence Grounding: Does the justification accurately reflect the provided evidence without hallucination?\n"
    "2. Numerical/Temporal Precision: Are all quantities, dates, and calculations correctly verified? Beware of the 'Numeric-Truth Effect'.\n"
    "3. Logical Consistency: Does the reasoning chain logically support the verdict?\n"
    "4. Verdict Alignment: Is the stated verdict consistent with the definitions above given the evidence?\n\n"
    "Output Rules:\n"
    "- Think step-by-step inside <think>... </think> tags.\n"
    "- After </think>, output EXACTLY one of the following formats:\n"
    "  • Yes, verdict is correct.\n"
    "  • No, verdict should be [Correct Verdict].\n"
    "- Replace [Correct Verdict] with True, False, or Conflicting.\n"
    "- Do not output any additional text after this line."
)

USER_TEMPLATE = (
    "### {input_text} \n\n" 
    "Audit this verdict and justification according to the criteria and definitions. "
    "Provide your reasoning in <think> tags, then output your final assessment in the exact required format."
)

class PushLoRACallback(TrainerCallback):
    def __init__(self, hf_token, base_repo_name, save_steps):
        self.hf_token = hf_token
        self.base_repo_name = base_repo_name
        self.save_steps = save_steps

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        # Push every `save_steps` steps
        if step > 0 and step % self.save_steps == 0:
            model = kwargs.get("model")
            tokenizer = kwargs.get("tokenizer")
            repo_id = f"{self.base_repo_name}-step-{step}"
            print(f"\n[PushLoRACallback] Pushing LoRA adapters to {repo_id}...")
            try:
                # With Unsloth/PEFT, push_to_hub automatically pushes only the adapters
                model.push_to_hub(repo_id, token=self.hf_token)
                if tokenizer:
                    tokenizer.push_to_hub(repo_id, token=self.hf_token)
                print(f"[PushLoRACallback] Successfully pushed to {repo_id}!\n")
            except Exception as e:
                print(f"\n[PushLoRACallback] Error pushing to Hub: {e}\n")

class GenerationLoggingCallback(TrainerCallback):
    def __init__(self, val_ds, log_path, generate_steps=200):
        self.val_ds = val_ds
        self.log_path = log_path
        self.generate_steps = generate_steps
        self.sample_messages = None
        if self.val_ds is not None and len(self.val_ds) > 0:
            # We want to keep system and user, drop assistant
            messages = self.val_ds[0]["messages"]
            self.sample_messages = [m for m in messages if m["role"] != "assistant"]
            
            # Clear log file initially
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("Generation Log initialized.\n===================================\n\n")

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step > 0 and step % self.generate_steps == 0 and self.sample_messages:
            model = kwargs.get("model")
            tokenizer = kwargs.get("tokenizer")
            if model and tokenizer:
                inputs = tokenizer.apply_chat_template(
                    self.sample_messages, 
                    tokenize=True, 
                    add_generation_prompt=True, 
                    return_tensors="pt"
                ).to(model.device)
                
                print(f"\n[GenerationLoggingCallback] Generating sample at step {step}...")
                
                # Generate
                outputs = model.generate(
                    inputs, 
                    max_new_tokens=512, 
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                )
                
                # Decode
                decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
                
                # Save to log file
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(f"========== STEP {step} ==========\n")
                    f.write(decoded_output)
                    f.write("\n===================================\n\n")


def format_dataset(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if config.get("run_config", {}).get("is_sanity", False):
        print(f"SANITY MODE ON: Truncating {dataset_path} to 2 items")
        data = data[:2]
        
    formatted_data = []
    for item in data:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(input_text=item["input_text"])},
            {"role": "assistant", "content": item["response"]}
        ]
        # By keeping the "messages" column, TRL SFTTrainer will automatically apply 
        # the chat template and properly compute `assistant_only_loss` masks!
        formatted_data.append({"messages": messages})
        
    return Dataset.from_pandas(pd.DataFrame(formatted_data))

def main():
    MODEL_ID = config["training"]["model_id"]
    TRAIN_PATH = config["paths"]["unified_sft_train"]
    VAL_PATH = config["paths"]["unified_sft_val"]
    OUTPUT_DIR = config["paths"]["output_dir"]
    GENERATION_LOG_PATH = config["paths"].get("generation_log", "./generation_log.log")
    BASE_REPO_NAME = f"{HF_USERNAME}/clef2026-qwen-lora"

    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID, 
        max_seq_length=config["training"]["max_seq_length"], 
        load_in_4bit=True,
        dtype=torch.bfloat16, 
        device_map="auto"
    )

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Formatting datasets for SFT...")
    train_ds = format_dataset(TRAIN_PATH)
    
    try:
        val_ds = format_dataset(VAL_PATH)
    except FileNotFoundError:
        print(f"[WARNING] Validation file not found at {VAL_PATH}. Proceeding without evaluation.")
        val_ds = None

    # Add LoRA
    print("Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model, 
        r=config["lora"]["r"], 
        lora_alpha=config["lora"]["alpha"], 
        lora_dropout=config["lora"]["dropout"], 
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth"
    )

    save_steps = config["training"].get("save_steps", 2000)

    callbacks_list = [PushLoRACallback(hf_token=HF_TOKEN, base_repo_name=BASE_REPO_NAME, save_steps=save_steps)]
    if val_ds is not None:
        callbacks_list.append(GenerationLoggingCallback(val_ds=val_ds, log_path=GENERATION_LOG_PATH, generate_steps=200))

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
            num_train_epochs=config["training"]["num_train_epochs"],
            learning_rate=float(config["training"]["learning_rate"]),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=config["training"]["logging_steps"],
            save_steps=save_steps,
            output_dir=OUTPUT_DIR,
            dataloader_num_workers=4,
            optim=config["training"]["optim"],
            max_seq_length=config["training"]["max_seq_length"],
            dataset_num_proc=4,
            packing=False,
            model_init_kwargs={"dtype": torch.bfloat16},
            assistant_only_loss=True,
        ),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=callbacks_list
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Training completed. Checkpoints saved to {OUTPUT_DIR}")
    
    # Optional final push
    try:
        print("Pushing final model to Hub...")
        model.push_to_hub(f"{BASE_REPO_NAME}-final", token=HF_TOKEN)
        tokenizer.push_to_hub(f"{BASE_REPO_NAME}-final", token=HF_TOKEN)
        print("Final push complete!")
    except Exception as e:
        print(f"Error on final push: {e}")

if __name__ == "__main__":
    main()
