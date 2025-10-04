import logging

from pymilvus import AsyncMilvusClient, MilvusException
 
logger = logging.getLogger(__name__)


class AsyncMilvusService:
    """
    An asynchronous Milvus client service implemented as a context manager 
    
    """
    
    def __init__(self, uri: str = "http://localhost:19530", token: str = None):
        """Initializes the AsyncMilvusClient instance."""
        self.uri = uri
        
        self.client = AsyncMilvusClient(uri=uri, token=token)
        logger.info("AsyncMilvusClient configured for: %s", self.uri)
    
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            await self.client.close()
            
        except MilvusException as milvus_err:
            logger.error("MilvusException occurred during client close: %s", str(milvus_err))
