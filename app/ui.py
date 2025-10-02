import streamlit as st
import requests

# --- Page Configuration ---
st.set_page_config(
    page_title="Project Soshyant",
    page_icon="🧠",
    layout="wide"
)

# --- UI Elements ---
st.title("🧠 Project Soshyant: Customer Insights Engine")
st.markdown("This PoC uses a RAG pipeline to answer questions about the Olist customer dataset.")

# User input
user_question = st.text_input(
    "Ask a question about a customer:",
    placeholder="e.g., What was the review for customer 861eff4711a542e4b93843c6dd7febb0?"
)

if st.button("Get Answer"):
    if user_question:
        with st.spinner("Retrieving information and generating answer..."):
            try:
                # --- API Call ---
                # This is where the UI talks to the FastAPI backend.
                api_url = "http://127.0.0.1:8000/ask"
                response = requests.post(api_url, json={"question": user_question})
                response.raise_for_status() # Raise an exception for bad status codes
                
                result = response.json()
                
                # --- Display Result ---
                st.subheader("Answer:")
                st.write(result['answer'])
                
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the API. Please ensure the backend is running. Error: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")