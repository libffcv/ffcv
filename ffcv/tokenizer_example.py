from transformers import AutoTokenizer

# Load a pre-trained tokenizer (BERT in this case)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Sample input text
text = ["Hello, how are you?", "I am learning NLP with Hugging Face!"]

# Tokenize text with padding
tokenized_output = tokenizer(
    text,
    padding=True,  # Apply padding
    truncation=True,  # Truncate if needed
    max_length=16,  # Set max length
    return_tensors="pt"  # Return PyTorch tensors
)

# Print tokenized output
print(tokenized_output)
