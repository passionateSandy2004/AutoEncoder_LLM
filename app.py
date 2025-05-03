import streamlit as st
import torch
import pickle
from QuickInference import answer_question

# Set page config
st.set_page_config(
    page_title="AE-LLM Question Answering",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stTextInput>div>div>input {
        background-color: #f0f2f6;
    }
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🤖 AE-LLM Question Answering")
st.markdown("""
This is an autoencoder-based language model that can answer questions based on its training data.
The model has been trained and fine-tuned on Wikipedia data.
""")

# Sidebar with model info
with st.sidebar:
    st.header("Model Information")
    st.markdown("""
    - **Model Type**: Autoencoder-based Language Model
    - **Training Data**: Wikipedia QA pairs
    - **Autoencoder Loss**: ~0.14
    - **Transition Network Loss**: <0.01
    """)
    
    st.header("Limitations")
    st.markdown("""
    - Limited to training domain knowledge
    - May not answer questions outside its training data
    - Requires careful fine-tuning to prevent forgetting
    """)

# Main content
st.header("Ask a Question")

# Text input for question
question = st.text_input(
    "Enter your question here:",
    placeholder="e.g., Which prize did Frederick Buechner create?"
)

# Submit button
if st.button("Get Answer"):
    if question:
        with st.spinner("Generating answer..."):
            try:
                # Get answer from model
                answer = answer_question(question)
                
                # Display answer
                st.markdown("### Answer:")
                st.info(answer)
                
                # Show example questions
                st.markdown("### Example Questions:")
                st.markdown("""
                - What does the Kroc Institute at Notre Dame focus on?
                - What institute at Notre Dame studies the reasons for violent conflict?
                """)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a question first!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit</p>
    <p>Model: AE-LLM | License: MIT</p>
</div>
""", unsafe_allow_html=True) 