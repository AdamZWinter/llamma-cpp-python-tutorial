#import json
from llama_cpp import Llama
import copy

#load the model
print("Loading the model...")
llm = Llama(model_path="../models/ggml-vic7b-q5_1.bin")
print("Model loaded.")
stream = llm(
    "Question: Who is Ada Lovelace? Answer:",
    max_tokens=100,
    temperature=0.9,
    stop=["\n", "Question:", "Q:"],
    stream=True
)


#print the output
#print(json.dumps(output, indent=2))
for out in stream:
    completionFragment = copy.deepcopy(out)
    print(completionFragment["choices"][0]["text"])

