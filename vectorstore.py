from langchain_chroma import Chroma
from langchain_qdrant import Qdrant 
from langchain_google_genai.google_vector_store import GoogleVectorStore

class VectorStore:
    def __init__(self, chunks):
        self.chunks = chunks

    def get_chroma_vectorstore(self, embedding_model):
        self.vectorstore = Chroma.from_documents(documents=self.chunks, embedding=embedding_model)
        return self.vectorstore
    def get_qdrant_vectorstore(self, embedding_model):
        self.vectorstore = Qdrant.from_documents(documents=self.chunks, embedding=embedding_model, location=":memory:")
        return self.vectorstore

    def get_google_vectorstore(self, embedding_model):
        self.vectorstore = GoogleVectorStore(corpus_id="123")
        self.vectorstore = self.vectorstore.from_documents(documents=self.chunks, embedding=embedding_model)
        return self.vectorstore

    def get_vector_store(self):
        return self.vectorstore
