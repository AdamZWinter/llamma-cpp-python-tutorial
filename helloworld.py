import json
from llama_cpp import Llama

#load the model
print("Loading the model...")
llm = Llama(model_path="../models/ggml-vic7b-q5_1.bin")
print("Model loaded.")
output = llm(
    "Question: Who is Ada Lovelace? Answer:",
    max_tokens=100,
    temperature=0.9,
    stop=["\n", "Question:", "Q:"],
    echo=True
)


#print the output
print(json.dumps(output, indent=2))