from langchain_chroma import Chroma
from langchain_qdrant import Qdrant 

class VectorStore:
    def __init__(self, chunks):
        self.chunks = chunks

    def create_vector_store(self, store_name, embedding_model):
        if store_name.lower() == "chroma":
            self.vectorstore = Chroma.from_documents(documents=self.chunks, embedding=embedding_model)
        elif store_name.lower() == "qdrant":
            self.vectorstore = Qdrant.from_documents(documents=self.chunks, embedding=embedding_model,location=":memory:")
        else:
            raise ValueError("Unsupported vector store type. Supported types: 'chroma', 'qdrant'.")

    def get_vector_store(self):
        return self.vectorstore
