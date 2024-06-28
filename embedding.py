from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class EmbeddingModel:
    def __init__(self, docs, embedding_model_name="BAAI/bge-small-en"):
        self.docs = docs

    def hugging_face_bge_embeddings(self, model_name="BAAI/bge-small-en"):
        self.model_name = model_name
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        sentence_transformer_ef = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return sentence_transformer_ef

    def fast_embed_embeddings(self, model_name="BAAI/bge-small-en-v1.5"):
        embeddings = FastEmbedEmbeddings(model_name=model_name)
        return embeddings

    def open_ai_embeddings(self, model_name="text-embedding-ada-002"):
        embeddings = OpenAIEmbeddings(model=model_name)
        return embeddings
    
    def google_embeddings(self,model_name = "models/embedding-001"):
        embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
        return embeddings
