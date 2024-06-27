import subprocess
import itertools
from tqdm import tqdm

# Define lists for each argument
#models = ["openai", "groq", "claude"]
#embeddings = ["openai", "hugging", "fast"]
#retrievers = ["base", "parent", "comp_extract", "comp_filter", "comp_emb"]
#pre_summarizes = [True, False]
#vectorstores = ["chroma", "qdrant"]

models = ["openai", "groq", "claude"]
embeddings = []
retrievers = ["parent","comp_emb"]
#pre_summarizes = [True, False]
vectorstores = ["chroma", "qdrant"]

# Define additional models for openai
additional_openai_models = ["gpt-4-turbo", "gpt-4o"]

# Define default values
default_model = "openai"
default_embedding = "fast"
default_retriever = "comp_emb"
default_pre_summarize = False
default_vectorstore = "qdrant"

# Use default value if list is empty
models = models or [default_model]
embeddings = embeddings or [default_embedding]
retrievers = retrievers or [default_retriever]
pre_summarizes = pre_summarizes or [default_pre_summarize]
vectorstores = vectorstores or [default_vectorstore]

# Generate all possible combinations of arguments
combinations = list(itertools.product(models, embeddings, retrievers, pre_summarizes, vectorstores))

def run_command(model, model_name, embedding, retriever, pre_summarize, vectorstore):
    command = [
        "python3", "main.py",
        "--files_path", "../FILES",
        "--model", model,
        "--model_name", model_name,
        "--embeddings", embedding,
        "--retriever", retriever,
        "--vectorstore", vectorstore
    ]
    if pre_summarize:
        command.append("--pre_summarize")
    result = subprocess.run(command, capture_output=True, text=True)
    print(f"Output for model {model}, model_name {model_name}, embedding {embedding}, retriever {retriever}, pre_summarize {pre_summarize}, vectorstore {vectorstore}:\n{result.stdout}")
    if result.stderr:
        print(f"Error for model {model}, model_name {model_name}, embedding {embedding}, retriever {retriever}, pre_summarize {pre_summarize}, vectorstore {vectorstore}:\n{result.stderr}")

# Run the command for each combination with progress bar
for combination in tqdm(combinations, desc="Running combinations", unit="combination"):
    model, embedding, retriever, pre_summarize, vectorstore = combination
    if model == "openai":
        for model_name in additional_openai_models:
            run_command(model, model_name, embedding, retriever, pre_summarize, vectorstore)
    else:
        run_command(model, model, embedding, retriever, pre_summarize, vectorstore)