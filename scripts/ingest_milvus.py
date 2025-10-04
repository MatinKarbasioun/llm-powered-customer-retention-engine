import argparse
import os
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from sentence_transformers import SentenceTransformer

from langchain.text_splitter import RecursiveCharacterTextSplitter

from pymilvus import MilvusClient, DataType


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

class Ingest:
    def __init__(self, milvus_host: str = "localhost", milvus_port: int = 19530):
        # 💡 MILVUS CONNECTION DETAILS
        self.milvus_uri = f"http://{milvus_host}:{milvus_port}"
        print(f"Milvus client initialized for URI: {self.milvus_uri}")

    def create_milvus_collection(self, collection_name: str, dim: int):
        """Initializes Milvus client and ensures the collection exists."""
        self.milvus_client = MilvusClient(uri=self.milvus_uri)

        if self.milvus_client.has_collection(collection_name):
            print(f"Collection '{collection_name}' already exists. Dropping and recreating...")
            self.milvus_client.drop_collection(collection_name)
        
        # 1. Define the collection schema
        schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=True, # Allows flexible metadata
        )

        # 2. Add fields: primary key, vector, and metadata fields
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="chunk_text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="customer_unique_id", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="order_id", datatype=DataType.VARCHAR, max_length=256)

        # 3. Prepare index parameters
        index_params = self.milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX", # AUTOINDEX is a good default choice
            metric_type="COSINE"
        )

        # 4. Create the collection
        self.milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        print(f"Successfully created Milvus collection '{collection_name}' with dimension {dim}.")
        return collection_name

    def insert_data_to_milvus(self, collection_name: str, documents_to_embed: list):
        """Inserts the processed chunks into the Milvus collection."""
        
        # Milvus uses a dict-list format for insertion, converting the list of dicts:
        # [{"embedding": [...], "chunk_text": "...", ...}, ...]
        
        # Milvus Client is already initialized in create_milvus_collection
        
        # Batch insertion is much faster
        BATCH_SIZE = 10000 
        
        for i in range(0, len(documents_to_embed), BATCH_SIZE):
            batch = documents_to_embed[i:i + BATCH_SIZE]
            
            # Milvus insertion function:
            self.milvus_client.insert(
                collection_name=collection_name,
                data=batch
            )
            print(f"Inserted {i + len(batch)}/{len(documents_to_embed)} chunks into Milvus.")

        # Ensure all data is visible immediately for searching
        self.milvus_client.flush(collection_name)
        print(f"Data ingestion complete. Total entities in '{collection_name}': {self.milvus_client.get_collection_stats(collection_name)['row_count']}")
        
    def create_vector_db(
        self,
        source_dir: str,
        text_column: str,
        collection_name: str, # 💡 Renamed from save_path to collection_name
        embedding_model: SentenceTransformer,
        chunk_size: int,
        chunk_overlap: int
    ): 
        # --- Chunking and Embedding Logic (Remains mostly the same) ---
        documents_to_embed = []
        df = pq.read_pandas(source_dir).to_pandas()
        text_splitter = self._get_text_splitter(chunk_size, chunk_overlap)
        
        print("\nStarting chunking and embedding process...")

        for index, row in df.iterrows():
            chunks = text_splitter.split_text(row[text_column])
            
            # Get the embedding vector dimension from the first chunk
            if not chunks:
                 continue
                 
            # NOTE: Milvus requires the embedding to be a Python list or numpy array
            chunk_embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
            
            # Store the results
            for i, chunk in enumerate(chunks):
                documents_to_embed.append({
                    # Note: Milvus will auto-generate the 'id' (primary key)
                    'customer_unique_id': row['customer_unique_id'],
                    'order_id': row['order_id'],
                    'chunk_text': chunk,
                    'embedding': chunk_embeddings[i] # Milvus calls the vector field 'embedding' here
                })
            
            if (index + 1) % 1000 == 0:
                print(f"Processed {index + 1}/{len(df)} documents...")

        print(f"\nGenerated a total of {len(documents_to_embed)} chunks.")
        
        # Get the dimension for the Milvus collection schema
        if not documents_to_embed:
            print("No documents to embed. Exiting.")
            return

        vector_dim = documents_to_embed[0]['embedding'].shape[0]

        # --- MILVUS INGESTION ---
        self.create_milvus_collection(collection_name, vector_dim)
        self.insert_data_to_milvus(collection_name, documents_to_embed)
        
    
    def _get_text_splitter(self, chunk_size: int = 1000, chunk_overlap: int=200):
        return RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
    
    def _get_embedding_model(self):
        # Ensure the model uses a CPU/GPU if available
        return SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')


def main():
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Create a Milvus vector database from DataFrame documents.")
    
    parser.add_argument(
        "--source",
        type=str,
        default="source_documents.parquet",
        help="The source PARQUET file containing the DataFrame."
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text_document",
        help="The name of text column in the DataFrame."
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="customer_data_collection",
        help="The name of the Milvus collection (table) to create."
    )
    parser.add_argument(
        "--milvus-host",
        type=str,
        default="localhost",
        help="The hostname or IP of the Milvus server."
    )
    parser.add_argument(
        "--milvus-port",
        type=int,
        default=19530,
        help="The port of the Milvus server."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="The maximum number of characters for each text chunk."
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="The number of characters to overlap between adjacent chunks."
    )
    
    args = parser.parse_args()
    
    ingest = Ingest(milvus_host=args.milvus_host, milvus_port=args.milvus_port)
    
    # Get the embedding model
    embedding_model = ingest._get_embedding_model()
    
    # Run the pipeline with the parsed arguments
    ingest.create_vector_db(
        source_dir=args.source,
        text_column=args.text_column,
        collection_name=args.collection_name,
        embedding_model=embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

if __name__ == "__main__":
    main()