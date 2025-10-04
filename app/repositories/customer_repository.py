from abc import ABC


class ICustomerRepository(ABC):
    
    @classmethod
    def get_retriever(cls, collection_name: str):
        """get retriever from vector store"""
        raise NotImplementedError("Subclasses must implement this method.")