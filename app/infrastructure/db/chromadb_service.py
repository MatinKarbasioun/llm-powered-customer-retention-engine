import logging
from typing import Optional

from chromadb import AsyncHttpClient, AsyncClientAPI


logger = logging.getLogger(__name__)

class AsyncChromaService:
    """
    An asynchronous ChromaDB client service implemented as a context manager.
    
    """

    def __init__(self, host: str = "localhost", port: int = 8000, token: Optional[str] = None):
        """Initializes the AsyncHttpClient instance."""
        self.host = host
        self.port = port
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.client: Optional[AsyncClientAPI] = None
        
        logger.info("AsyncChromaService configured for: http://%s:%s", self.host, self.port)

    
    async def __aenter__(self) -> 'AsyncChromaService':

        self.client = AsyncHttpClient(
            host=self.host,
            port=self.port,
            headers=self.headers
        )
        
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            try:
                if hasattr(self.client, 'close'):
                    await self.client.close()
                
                logger.debug("AsyncChromaService connection closed automatically.")
                
            except Exception as e:
                logger.error("Error occurred during Chroma client close: %s", str(e))
