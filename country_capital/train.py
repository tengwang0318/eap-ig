from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import TrainerCallback
import argparse
import deepspeed
import torch
import numpy as np


# Load Data
def load_data(tokenizer, train_filepath='country_capital_train.jsonl', test_filepath='country_capital_test.jsonl'):
    dataset = load_dataset("json", data_files={"train": train_filepath, "test": test_filepath})

    def tokenize_function(examples):
        return tokenizer(examples['input'], truncation=True, padding=True, max_length=24)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets


# Load Model
def load_model(model_name="EleutherAI/pythia-1.4b-deduped"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation='flash_attention_2',  # Comment this out if you don't need flash attention 2
        torch_dtype=torch.float16
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


class CacheCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is not None:
            model.config.use_cache = True  # Enable cache during evaluation

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            model.config.use_cache = False  # Disable cache during training

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            model.config.use_cache = False  # Disable cache at the end of training


# Custom Evaluation Function
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    # Convert logits to predictions
    generated_tokens = np.argmax(logits[:, 0, :], axis=-1)  # Only consider the first token
    # Compare with the first token of the labels
    correct = (generated_tokens == labels[:, 0]).astype(int)
    accuracy = np.mean(correct)
    return {"accuracy": accuracy}


# Main Function
if __name__ == '__main__':
    # Parse Command-Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/home/data/pythia-1.4b-deduped/")
    parser.add_argument('--train_filepath', type=str, default='data_correct_logic/country_capital_train.jsonl')
    parser.add_argument('--test_filepath', type=str, default='data_correct_logic/country_capital_test.jsonl')
    args = parser.parse_args()

    # Load Model and Tokenizer
    model, tokenizer = load_model(args.model_name)

    # Load and Tokenize Datasets
    datasets = load_data(tokenizer, train_filepath=args.train_filepath, test_filepath=args.test_filepath)

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        learning_rate=5e-5,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        report_to="tensorboard",
        deepspeed="ds_config.json",
        fp16=True,
        predict_with_generate=True  # Use generation during evaluation
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['test'],
        compute_metrics=compute_metrics,
        callbacks=[CacheCallback()],  # Pass the custom CacheCallback here
    )

    # Train the Model
    trainer.train()
