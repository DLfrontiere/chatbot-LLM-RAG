import argparse
from loading import Loader
from embedding import EmbeddingModel
from retrieving import BaseRetriever, ParentRetriever, CompressionExtractorRetriever, CompressionFilterRetriever, CompressionEmbeddingRetriever
from answer_generation import AnswerGenerator
from gui import GUI
from dotenv import load_dotenv
from document_processing import DocumentProcessor
from pathlib import Path
from models import OpenAIModel, GroqModel, ClaudeModel
from vectorstore import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import create_or_update_csv
import time

load_dotenv(Path("../api_key.env"))

def main():
    parser = argparse.ArgumentParser(description="Choose model, embeddings, retriever, and other options.")
    parser.add_argument('--model', choices=['openai', 'groq', 'claude'], default='openai', help="Choose the model to use (default: openai).")
    
    # Parse known args first to determine the model
    known_args, remaining_args = parser.parse_known_args()

        # Set default values based on the known_args
    if known_args.model == 'openai':
        default_model_name = 'gpt-3.5-turbo'
    elif known_args.model == 'groq':
        default_model_name = 'llama3-70b-8192'
    elif known_args.model == 'claude':
        default_model_name = 'claude-3-sonnet-20240229'
    
    parser.add_argument('--embeddings', choices=['openai', 'hugging', 'fast'], default='fast', help="Choose the embeddings to use.")
    parser.add_argument('--retriever', choices=['base', 'parent', 'comp_extract', 'comp_filter', 'comp_emb'], default='parent', help="Choose the retriever to use.")
    parser.add_argument('--files_path', type=str, required=True, help="Path to the directory containing files to be retrieved.")
    parser.add_argument('--pre_summarize', action='store_true', help="Whether to pre-summarize the documents (default: False).")
    parser.add_argument('--vectorstore', choices=['chroma', 'qdrant'], default='qdrant', help="Choose the vector store to use (default: qdrant).")
    parser.add_argument('--model_name', type=str, default=default_model_name, help="Model name based on the chosen model.")

    # Parse all args including the remaining args
    args = parser.parse_args(remaining_args)

    files_path = args.files_path
    accepted_files = ["pdf", "txt", "html","docx","doc"]
    urls = ["https://ainews.it/synthesia-creazione-di-avatar-ai-anche-da-mobile/"]

    # Choose model
    model_name = args.model_name
    if args.model == 'openai':
        model = OpenAIModel(model_name=model_name)
        llm = model.get_model()

    elif args.model == 'groq':
        model = GroqModel(model_name=model_name)
        llm = model.get_model()

    elif args.model == 'claude':
        model = ClaudeModel(model_name=model_name)
        llm = model.get_model()

    loader = Loader(files_path)
    docs_urls = loader.load_urls(urls)
    docs = loader.load_documents(accepted_files)
    docs.extend(docs_urls)

    # Optionally pre-summarize documents
    if args.pre_summarize:
        doc_processing = DocumentProcessor(docs, llm)
        docs = doc_processing.summarize_docs(docs)

    embedding_model = EmbeddingModel(docs)

    # Choose embedding function
    if args.embeddings == 'openai':
        embedding_function = embedding_model.open_ai_embeddings()
    elif args.embeddings == 'hugging':
        embedding_function = embedding_model.hugging_face_bge_embeddings()
    elif args.embeddings == 'fast':
        embedding_function = embedding_model.fast_embed_embeddings()

    base_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20, add_start_index=True)
    parent_splitter =  RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80, add_start_index=True)
    child_splitter = base_splitter
    chunks = base_splitter.split_documents(docs)
    
    # Initialize the vector store based on the chosen option
    vector_store_chunks = VectorStore(chunks)
    vector_store_chunks.create_vector_store(args.vectorstore, embedding_function)
    vectorstore_chunks = vector_store_chunks.get_vector_store()

    vector_store_docs = VectorStore(docs)
    vector_store_docs.create_vector_store(args.vectorstore, embedding_function)
    vectorstore_docs = vector_store_docs.get_vector_store()

    # Choose retriever
    if args.retriever == 'base':
        chosen_retriever = BaseRetriever(vectorstore_chunks).get_retriever()
    elif args.retriever == 'parent':
        chosen_retriever = ParentRetriever(docs, vectorstore_docs,parent_splitter,child_splitter).get_retriever()
    elif args.retriever == 'comp_extract':
        base_retriever = BaseRetriever(vectorstore_chunks).get_retriever()
        chosen_retriever = CompressionExtractorRetriever(base_retriever, llm).get_retriever()
    elif args.retriever == 'comp_filter':
        base_retriever = BaseRetriever(vectorstore_chunks).get_retriever()
        chosen_retriever = CompressionFilterRetriever(base_retriever, llm).get_retriever()
    elif args.retriever == 'comp_emb':
        base_retriever = BaseRetriever(vectorstore_chunks).get_retriever()
        chosen_retriever = CompressionEmbeddingRetriever(base_retriever,embedding_function=embedding_function).get_retriever()

    answer_generator = AnswerGenerator(retriever= chosen_retriever,model = llm)

    #GUI(answer_generator)

    prompts = ["what is nvidia culitho?","what's the washing machine name?","how much is claude 3.5 sonnet plan?","why Nvidia don't use org-charts?","what not to do to move the washing machine?","what's the next step in the broader vision of Claude.ai?"]
    groundthruts = ["NVIDIA cuLitho,a new library that supercharges computational lithography, an immensec omputational workload in chip design and manufacturing.","the washing machine name is Dyson Contrarotator","Claude 3.5 Sonnet is now available for free on Claude.ai and the Claude iOS app, while Claude Pro and Team plan subscribers can access it with significantly higher rate limits. It is also available via the Anthropic API, Amazon Bedrock, and Google Cloud’s Vertex AI. The model costs $3 per million input tokens and $15 per million output tokens, with a 200K token context window.","Nvidia doesn't use org-charts because they believe the mission is the boss","Do not push the washing machine with your foot","Claude.ai next step is to expand to support team collaboration"]

    for i,prompt in enumerate(prompts):
        print("Answering: ",prompt)
        groundtruth = groundthruts[i]
        context = answer_generator.get_current_context(prompt)
        start_time = time.time()  # Start the timer
        answer = answer_generator.answer_prompt(prompt)
        answer_time = round ( time.time() - start_time , 3)  # Calculate answer time
        create_or_update_csv(prompt, answer,groundtruth, context,answer_time, model_name, args.embeddings, args.retriever, args.pre_summarize, args.vectorstore, csv_file="./model_test.csv")
        print("chatbot: ",answer)


    
if __name__ == "__main__":
    main()
