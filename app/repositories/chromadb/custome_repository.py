from ..customer_repository import ICustomerRepository


class ChromaDBCustomerRepository(ICustomerRepository):
    def __init__(self, embedding_fucn: SentenceTransformerEmbeddings):
        self._embedding_func = embedding_fucn
        self._retriever = self.get_retriever()
        
    def create_collection(self, collection_name: str):
        """create a collection and ensures the collection exists."""
        
    
    def get_retriever(self, collection_name: str):
        """get retriever from vector store"""
 
        vector_store = Chroma(
            persist_directory=DB_DIR,
            collection_name=collection_name,
            embedding_function=self._embedding_func
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        return retriever
    
    @property
    def retriever(self):
        return self._retriever