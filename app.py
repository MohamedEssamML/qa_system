import streamlit as st
from inference import load_model, answer_question

st.title("BERT Question-Answering System")

# Load model
@st.cache_resource
def get_model():
    return load_model('models/bert-finetuned-squad')

model, tokenizer, device = get_model()

# Input fields
context = st.text_area("Enter the context:", height=200)
question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if context and question:
        answer = answer_question(question, context, model, tokenizer, device)
        st.write("**Answer:**")
        st.write(answer)
    else:
        st.error("Please provide both context and question.")