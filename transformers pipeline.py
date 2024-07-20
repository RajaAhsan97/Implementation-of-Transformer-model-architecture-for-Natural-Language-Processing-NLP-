"""
    Text generation using Transformers pipeline by Hugging Face library 
"""

from transformers import pipeline

# default model for transformers pipeline {openai-community/gpt2}
text_generator = pipeline(model="openai-community/gpt2")

prompt = "I am using Tranformers Model for the purpose of text-generation"
print(len(prompt.split()))

sequences = text_generator(prompt, num_return_sequences=3, max_length=30) 
print("Generated Text:")

for sequence in sequences:
    print(sequence["generated_text"])
    print("-------------------------")
    print("-------------------------")
