from app.services import RagChain


class RagChainDependency:
    def __init__(self):
        self._rag_chain_instance = RagChain()

    def __call__(self):
        return self._rag_chain_instance.chain
