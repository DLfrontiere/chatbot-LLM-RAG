import argparse
from loading import Loader
from embedding import EmbeddingModel
from retrieving import Retriever, ParentRetriever, CompressionExtractorRetriever, CompressionFilterRetriever, CompressionEmbeddingRetriever
from answer_generation import AnswerGenerator
from gui import GUI
import os
from dotenv import load_dotenv
from document_processing import DocumentProcessor
from pathlib import Path
from models import OpenAIModel, GroqModel, ClaudeModel

load_dotenv(Path("../api_key.env"))

def main():
    parser = argparse.ArgumentParser(description="Choose model, embeddings, retriever, and other options.")
    parser.add_argument('--model', choices=['openai', 'groq', 'claude'], default='openai', help="Choose the model to use (default: openai).")
    parser.add_argument('--embeddings', choices=['openai', 'hugging', 'fast'], default='fast', help="Choose the embeddings to use (default: fast).")
    parser.add_argument('--retriever', choices=['base', 'parent', 'comp_extract', 'comp_filter', 'comp_emb'], default='comp_emb', help="Choose the retriever to use (default: comp_emb).")
    parser.add_argument('--files_path', type=str, required=True, help="Path to the directory containing files to be retrieved.")
    parser.add_argument('--pre_summarize', action='store_true', help="Whether to pre-summarize the documents (default: False).")
    
    args = parser.parse_args()

    files_path = args.files_path
    accepted_files = ["pdf", "txt", "html","docx","doc"]
    urls = ["https://ainews.it/synthesia-creazione-di-avatar-ai-anche-da-mobile/"]

    # Choose model
    if args.model == 'openai':
        model = OpenAIModel().get_model()
    elif args.model == 'groq':
        model = GroqModel().get_model()
    elif args.model == 'claude':
        model = ClaudeModel().get_model()

    loader = Loader(files_path)
    docs_urls = loader.load_urls(urls)
    docs = loader.load_documents(accepted_files)
    docs.extend(docs_urls)

    # Optionally pre-summarize documents
    if args.pre_summarize:
        doc_processing = DocumentProcessor(docs, model)
        docs = doc_processing.summarize_docs(docs)

    embedding_model = EmbeddingModel(docs)
    
    # Choose embedding function
    if args.embeddings == 'openai':
        embedding_function = embedding_model.open_ai_embeddings()
    elif args.embeddings == 'hugging':
        embedding_function = embedding_model.hugging_face_bge_embeddings()
    elif args.embeddings == 'fast':
        embedding_function = embedding_model.fast_embed_embeddings()

    embedding_model.create_vector_store(docs, embedding_function)
    vectorstore = embedding_model.get_vector_store()

    # Choose retriever
    if args.retriever == 'base':
        chosen_retriever = Retriever(docs, embedding_function).get_retriever()
    elif args.retriever == 'parent':
        chosen_retriever = ParentRetriever(docs, vectorstore).get_retriever()
    elif args.retriever == 'comp_extract':
        base_retriever = Retriever(docs, embedding_function).get_retriever()
        chosen_retriever = CompressionExtractorRetriever(base_retriever, model).get_retriever()
    elif args.retriever == 'comp_filter':
        base_retriever = Retriever(docs, embedding_function).get_retriever()
        chosen_retriever = CompressionFilterRetriever(base_retriever, model).get_retriever()
    elif args.retriever == 'comp_emb':
        base_retriever = Retriever(docs, embedding_function).get_retriever()
        chosen_retriever = CompressionEmbeddingRetriever(base_retriever, docs=docs, embedding_function=embedding_function).get_retriever()

    answer_generator = AnswerGenerator(retriever= chosen_retriever,model = model)
    GUI(answer_generator)
    
if __name__ == "__main__":
    main()
