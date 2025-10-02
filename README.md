# LLM-Powered Customer Retention Engine (LCRE)
This repository contains the proof-of-concept (PoC) for an LLM-powered customer retention engine. The goal of this PoC is to build and validate an end-to-end system using a Retrieval-Augmented Generation (RAG) framework on a public dataset as proof step before production-ready product

## 🎯 Objective

The primary objective is to answer complex, natural language questions about customers by synthesizing information from both unstructured text (product reviews) and structured data (order history). This will prove the viability of the architecture for deriving deep customer insights.

**Example Query:** "What are the main complaints from customer `{customer_id}` and what was their last order?"

## 📦 Dataset

This LCRE uses the **Brazilian E-Commerce Public Dataset by Olist**, available on Kaggle. This dataset was chosen because it effectively simulates our target data structure:
* **Customer Reviews:** Simulates unstructured, multilingual customer support conversations.
* **Order & Customer Data:** Simulates structured trade and CRM data.

## 🏗️ Architecture

The system is built on a RAG (Retrieval-Augmented Generation) architecture:

1.  **Data Processing:** Raw CSV data is cleaned, transformed into natural language sentences, and split into manageable chunks.
2.  **Vectorization:** Each text chunk is converted into a numerical vector (embedding) using a sentence-transformer model.
3.  **Indexing:** The text chunks and their corresponding vectors are stored in a **ChromaDB** vector database.
4.  **Retrieval:** When a user asks a question, the system retrieves the most relevant chunks from ChromaDB.
5.  **Generation:** The user's question and the retrieved chunks are passed to a Large Language Model (LLM), which generates a synthesized, human-readable answer.

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* Git

### Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MatinKarbasioun/llm-powered-customer-retention-engine
    cd llm-powered-customer-retention-engine
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the data:**
    Download the Olist dataset from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place the relevant CSV files (`olist_order_reviews_dataset.csv`, `olist_orders_dataset.csv`, `olist_customers_dataset.csv`) into the `data/` directory.

5. **Create /db directory:**
    To store ChromaDB persistence vector, create this directory as the place to located ChromaDB database.

    ```bash
    mkdir /db
    ```
5.  **Set up environment variables:**
    You will need an API key for the LLM provider (e.g., OpenAI, HuggingFace, and Kaggle). Create a `.env` file in the root directory, copy everything from `example.env` or rename it and replace the OpenAI API key, HuggingFace Token and Kaggle username and keu with your own.

    ```
    OPENAI_API_KEY="your_api_key_here"
    HF_TOKEN="your_hf_token_here"
    KAGGLE_USERNAME="your_kaggle_username_here"
    KAGGLE_KEY="your_kaggle_key_here"
    ```

6.  **Run the FastAPI application:**
    
    ```bash
     uvicorn app.main:app --reload

    ```

7. **Run the Streamlit UI:**

    ```bash
    streamlit run app.py
    
   ```


