import argparse
import os

import pandas as pd
import pickle
import pyarrow.parquet as pq

from sentence_transformers import SentenceTransformer


from langchain.text_splitter import RecursiveCharacterTextSplitter

class Ingest:   
    def __init__(self):
        pass
    
    def create_vector_db(
        self,
        source_dir: str,
        text_column: str,
        save_path: str,
        save_chunked_data_path: str,
        embedding_model: SentenceTransformer,
        chunk_size: int,
        chunk_overlap: int
    ):  
        # Chunking
        documents_to_embed = []

        print("\nStarting chunking and embedding process...")
        
        df = pq.read_pandas(source_dir).to_pandas()
        
        text_splitter = self._get_text_splitter(chunk_size, chunk_overlap)
        

        for index, row in df.iterrows():
            chunks = text_splitter.split_text(row[text_column])
            
            # Generate an embedding for each chunk
            chunk_embeddings = embedding_model.encode(chunks)
            
            # Store the results
            for i, chunk in enumerate(chunks):
                documents_to_embed.append({
                    'customer_unique_id': row['customer_unique_id'],
                    'order_id': row['order_id'],
                    'chunk_text': chunk,
                    'embedding': chunk_embeddings[i]
                })
            
            if (index + 1) % 1000 == 0:
                print(f"Processed {index + 1}/{len(df)} documents...")

        print(f"\nGenerated a total of {len(documents_to_embed)} chunks.")

        # --- 5. Inspect and Save ---
        print("\n--- Example of a single processed document ---")
        example = documents_to_embed[0]
        print("Customer ID:", example['customer_unique_id'])
        print("Chunk Text:", example['chunk_text'])
        print("Embedding Shape:", example['embedding'].shape) # Will be (768,) for this model

        # Save the final list of chunks and embeddings to a file
        with open(save_path, 'wb') as f:
            pickle.dump(documents_to_embed, f)
            
        print(f"\nChunks and embeddings saved to '{save_path}'.")
    
    def _get_text_splitter(self, chunk_size: int = 1000, chunk_overlap: int=200):
        return RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
    
    def _get_embedding_model(self):
        return SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
    
if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Create a vector database from PDF documents.")
    
    parser.add_argument(
        "--source",
        type=str,
        default="source_documents",
        help="The source directory containing the DataFrame files."
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text_document",
        help="The name of text column in the DataFrame."
    )
    parser.add_argument(
        "--save-chunked_data_path",
        type=str,
        default="/db",
        help="The path where the vector db persists."
    )
    parser.add_argument(
    "--save-path",
    type=str,
    default="/db",
    help="The path where the vector db persists."
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
    
    # Run the pipeline with the parsed arguments
    create_vector_db(
        source_dir=args.source,
        text_column=args.text_column,
        save_path=args.save_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    
