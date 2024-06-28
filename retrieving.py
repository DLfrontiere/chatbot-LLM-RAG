from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.retrievers import ParentDocumentRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, EmbeddingsFilter
from langchain.storage import InMemoryStore
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers.multi_query import MultiQueryRetriever

#check compliance between base retriever and compression retrievers
class BaseRetriever:

    def __init__(self,vectorstore):
        self.vectorstore = vectorstore

    def get_retriever(self):
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 5})
        return retriever
    
class ParentRetriever:

    def __init__(self, docs, vectorstore, parent_splitter=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=75, add_start_index=True), child_splitter=RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=15, add_start_index=True)):
        self.docs = docs
        self.vectorstore = vectorstore
        self.child_splitter = parent_splitter
        self.parent_splitter = child_splitter

    def get_retriever(self):
        docs = self.docs
        store = InMemoryStore()
        retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )

        retriever.add_documents(docs)
        return retriever

class MultiQueryDataRetriever(BaseRetriever):

    def __init__(self, base_retriever,model):
        self.base_retriever = base_retriever
        self.model = model

    def get_multi_retriever(self):
        llm = self.model
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=self.base_retriever, llm=llm
        )
        return self.retriever
    
    def get_retriever(self):
        return self.get_multi_retriever()
    
class CompressionExtractorRetriever:

    def __init__(self, base_retriever,model):
        self.base_retriever = base_retriever
        self.model = model

    def get_compression_retriever(self):
        llm = self.model
        compressor = LLMChainExtractor.from_llm(llm)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.base_retriever
        )
        return self.retriever
    
    def get_retriever(self):
        return self.get_compression_retriever()

class CompressionFilterRetriever:

    def __init__(self, base_retriever,model):
        self.base_retriever = base_retriever
        self.model = model

    def get_compression_retriever(self):
        llm = self.model
        compressor = LLMChainFilter.from_llm(llm)
        compression_retriever = compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.base_retriever
        )
        return compression_retriever

    def get_retriever(self):
        return self.get_compression_retriever()

class CompressionEmbeddingRetriever:

    def __init__(self, base_retriever,embedding_function):
        self.embedding_function = embedding_function
        self.base_retriever = base_retriever

    def get_compression_embedding_retriever(self):

        embeddings_filter = EmbeddingsFilter(embeddings=self.embedding_function, similarity_threshold=0.76)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=embeddings_filter, base_retriever=self.base_retriever
        )
        return compression_retriever

    def get_retriever(self):
        return self.get_compression_embedding_retriever()


