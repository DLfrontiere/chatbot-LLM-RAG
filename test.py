import subprocess
import itertools

# Define lists for each argument
models = ["openai", "groq", "claude"]
embeddings = ["openai", "hugging", "fast"]
retrievers = ["base", "parent", "comp_extract", "comp_filter", "comp_emb"]
pre_summarizes = [True, False]
vectorstores = ["chroma", "qdrant"]

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
combinations = itertools.product(models, embeddings, retrievers, pre_summarizes, vectorstores)

def run_command(model, embedding, retriever, pre_summarize, vectorstore):
    command = [
        "python3", "main.py",
        "--files_path", "../FILES",
        "--model", model,
        "--embeddings", embedding,
        "--retriever", retriever,
        "--vectorstore", vectorstore
    ]
    if pre_summarize:
        command.append("--pre_summarize")
    result = subprocess.run(command, capture_output=True, text=True)
    print(f"Output for model {model}, embedding {embedding}, retriever {retriever}, pre_summarize {pre_summarize}, vectorstore {vectorstore}:\n{res


