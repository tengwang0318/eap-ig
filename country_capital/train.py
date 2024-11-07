import json
import os

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import TrainerCallback, DataCollatorForLanguageModeling
import argparse
import deepspeed
import torch
import numpy as np

LENGTHS = []


#
def load_data(tokenizer, train_filepath='country_capital_train.jsonl', test_filepath='country_capital_test.jsonl'):
    dataset = load_dataset("json", data_files={"train": train_filepath, "test": test_filepath})

    def tokenize_function(examples):
        # Ġ中间加了个
        tokenizer_origin_input = tokenizer(
            examples['clean']
        )

        lengths = [len(item) for item in tokenizer_origin_input['input_ids']]

        combined_texts = [f"{clean} {label}" for clean, label in zip(examples['clean'], examples['label'])]
        tokenized_inputs = tokenizer(
            combined_texts,
            truncation=True,
            padding=True,
            max_length=28
        )
        labels = [input_ids[length] for length, input_ids in zip(lengths, tokenized_inputs['input_ids'])]

        tokenized_inputs['label'] = torch.tensor(labels, dtype=torch.long)
        # print(combined_texts[0])
        # print(tokenized_inputs["input_ids"][0])
        # print(tokenized_inputs["label"][0])
        # os.abort()
        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": tokenized_inputs["label"],
        }

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["clean", "corrupt", "label", "corrupt_label"]
    )
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


def compute_metrics(eval_preds):
    logits, labels = eval_preds

    predicted_tokens = np.argmax(logits, axis=-1)  # (batch_size, sequence_length)

    global LENGTHS
    batch_size = logits.shape[0]

    if len(LENGTHS) != batch_size:
        raise ValueError(f"Expected LENGTHS to have size {batch_size}, but got {len(LENGTHS)}.")

    generated_tokens = np.array([predicted_tokens[i, LENGTHS[i]] for i in range(batch_size)])
    ground_truth = np.array([labels[i, LENGTHS[i]] for i in range(batch_size)])
    print(predicted_tokens.shape)
    print(labels.shape)
    print(generated_tokens.shape)

    correct = (generated_tokens == ground_truth).astype(int)
    accuracy = np.mean(correct)

    print("Generated Tokens:", generated_tokens)
    print("ground truth:", ground_truth)
    print("Correct Predictions:", correct)
    print("Accuracy:", accuracy)

    return {"accuracy": accuracy}


def load_all_test_data(test_path, tokenizer):
    with open(test_path) as f:
        for line in f.readlines():
            dic = json.loads(line)
            input_ids = tokenizer(dic['clean'])['input_ids']
            LENGTHS.append(len(input_ids))
    print(f"Length: {len(LENGTHS)}")


# Main Function
if __name__ == '__main__':
    # Parse Command-Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-1.4b-deduped")
    parser.add_argument('--train_filepath', type=str, default='data_correct_logic/country_capital_train.jsonl')
    parser.add_argument('--test_filepath', type=str, default='data_correct_logic/country_capital_test.jsonl')
    args = parser.parse_args()

    # Load Model and Tokenizer
    model, tokenizer = load_model(args.model_name)

    # Load and Tokenize Datasets
    datasets = load_data(tokenizer, train_filepath=args.train_filepath, test_filepath=args.test_filepath)
    load_all_test_data(args.test_filepath, tokenizer)
    for batch in datasets['train']:
        print(batch)
        break
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=1,
        save_strategy="epoch",
        save_total_limit=3,
        learning_rate=5e-5,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        report_to="tensorboard",
        deepspeed="ds_config.json",
        fp16=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=datasets['train'],
        eval_dataset=datasets['test'],
        compute_metrics=compute_metrics,
        callbacks=[CacheCallback()],
    )

    # Train the Model
    trainer.train()
