import torch
from torch.utils.data import DataLoader
from train import load_model, load_data  # Assuming these functions are defined in train.py

# Function to Evaluate Model with Batch Size and Compute Accuracy
def evaluate(tokenizer, model, test_dataset, batch_size=8, max_new_tokens=1):
    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare the DataLoader
    dataloader = DataLoader(test_dataset, batch_size=batch_size)

    total_samples = 0
    correct_predictions = 0
    results = []

    with torch.no_grad():
        for batch in dataloader:
            # Extract input IDs and labels, and move them to the device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)  # Assuming labels are available in the dataset

            # Generate output
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,  # Only generate the first token
                do_sample=False,  # Use greedy decoding
                eos_token_id=tokenizer.eos_token_id  # End-of-sequence token
            )

            # Compare generated tokens with the true labels
            # Get the first token of the generated output
            generated_tokens = output_ids[:, 0]
            true_tokens = labels[:, 0]

            # Calculate the number of correct predictions
            correct_predictions += (generated_tokens == true_tokens).sum().item()
            total_samples += len(true_tokens)

            # Decode the input and output for display
            for i in range(len(input_ids)):
                input_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                generated_text = tokenizer.decode(output_ids[i], skip_special_tokens=True)
                results.append({"input": input_text, "output": generated_text})

    # Calculate accuracy
    accuracy = correct_predictions / total_samples
    print(f"Accuracy: {accuracy:.4f}")

    return results

if __name__ == '__main__':
    # Load Model and Tokenizer
    model_name = "EleutherAI/pythia-1.4b-deduped"
    model, tokenizer = load_model(model_name)

    # Load the Tokenized Test Dataset
    datasets = load_data(tokenizer)
    test_dataset = datasets['test']  # Use the test dataset for evaluation

    # Evaluate with Batch Size
    batch_size = 4
    results = evaluate(tokenizer, model, test_dataset, batch_size=batch_size)

    # Print Results
    for result in results:
        print(f"Input: {result['input']}")
        print(f"Output: {result['output']}")
        print()
