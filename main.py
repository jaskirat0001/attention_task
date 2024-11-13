from dotenv import load_dotenv
import os
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib
import requests
import sqlite3
 
matplotlib.use('TkAgg')


load_dotenv()

st.title("Lets Start the Research")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# # Load the model and tokenizer
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
def chatbot_response(query):
    # Tokenize input
    inputs = tokenizer.encode(query, return_tensors="pt")

    # Generate response
    outputs = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
# First Section we will take the input from the User so that he can enter the topic related to what rsearch papers are needed to be found
query = st.text_input("Enter the Area in which you are interested to explore")
if query:
    st.write(f"Searching for papers related to '{query}'...")

st.title("Open Question-Answer Chatbot")

# Input text box
user_query = st.text_input("Ask me anything:")

if user_query:
    # Generate chatbot response
    response = chatbot_response(user_query)
    st.write(f"Chatbot: {response}")

def extract_field(entry, start_tag, end_tag):
    """Extract content from an XML entry based on start and end tags."""
    start_idx = entry.find(start_tag)
    if start_idx == -1:
        return ""  # Return empty string if start tag is not found
    start_idx += len(start_tag)  # Move to the end of the start tag
    end_idx = entry.find(end_tag, start_idx)
    if end_idx == -1:
        return ""  # Return empty string if end tag is not found
    return entry[start_idx:end_idx]


def fetch_arxiv_papers(query, start_year=2019, max_results=50):
    base_url = "http://export.arxiv.org/api/query?"
    query_params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    response = requests.get(base_url, params=query_params)
    
    if response.status_code != 200:
        st.error("Error fetching data from arXiv API")
        return []

    papers = []
    entries = response.text.split("<entry>")[1:]  # Split by entry to get each paper

    for entry in entries:
        # Extract relevant fields (title, published date, summary, authors)
        title = extract_field(entry, "<title>", "</title>")
        published_date = extract_field(entry, "<published>", "</published>")
        summary = extract_field(entry, "<summary>", "</summary>")
        link = extract_field(entry, "<id>", "</id>")
        
        # Convert date and filter by year
        published_year = int(published_date[:4])
        if published_year >= start_year:
            papers.append({
                "title": title.strip(),
                "published_date": published_date,
                "summary": summary.strip(),
                "link": link.strip()
            })
    
    return papers



# Function to get the response from the chatbot using Ollama's API
# def answer_question(question, context):
#     prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
#     response = ollama.invoke(model=model_name, prompt=prompt)
#     return response.strip()


if query:
    papers = fetch_arxiv_papers(query)
    
    if papers:
        st.write("Papers Found:")
        
        # Connect to SQLite database
        conn = sqlite3.connect("arxiv_papers.db")
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                published_date TEXT,
                summary TEXT,
                link TEXT
            )
        """)

        # Display and insert each paper into the database
        for paper in papers:
            st.subheader(paper["title"])
            st.write(f"Published: {paper['published_date']}")
            st.write(f"Summary: {paper['summary']}")
            st.write(f"Link: [Read more]({paper['link']})")

            # Insert into the database
            cursor.execute("""
                INSERT INTO Papers (title, published_date, summary, link) VALUES (?, ?, ?, ?)
            """, (paper["title"], paper["published_date"], paper["summary"], paper["link"]))

        # Commit changes and close the database connection
        conn.commit()
        conn.close()
        st.success("Papers saved to database.")




# section 3
# Now we are creating a Question and Answer Chatbot that can answer the user queries using the LLAMA model
# from ollama import Ollama
# model = Ollama.load_model("llama")

# def chatbot_response(user_input):
#     response = model.generate(user_input,max_length=100,temperature=0.7)
#     return response.text



# uploaded_file = st.file_uploader("Upload a csv file for analysis", type=["csv"])

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     pandas_ai = SmartDataframe(df)
#     # st.write(df)
#     st.write(df.head())
    
#     prompt = st.text_input("Enter a prompt")
#     if st.button("Generate"):
#         if prompt:
#             st.write("PandasAI is generating a response...")
#             st.write(pandas_ai.chat(prompt))
#         else:
#             st.warning("Please enter a prompt")






# # # from vllm import LLM,SamplingParams
# # # model_name = "llama"
# # # llm = LLM(model = model_name)

# def chatbot_response(query):
#     # Tokenize input
#     inputs = tokenizer.encode(query, return_tensors="pt")

#     # Generate response
#     outputs = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id, temperature=0.7)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response
# model_name = "microsoft/MiniLM-L6-H384-uncased"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Define the chatbot response function
# def generate_response(question):
#     # Tokenize the input question
#     inputs = tokenizer(question, return_tensors="pt")
#     # Generate a response from the model
#     outputs = model.generate(
#         inputs["input_ids"], 
#         max_length=100, 
#         temperature=0.7, 
#         top_p=0.9, 
#         do_sample=True
#     )
#     # Decode the generated tokens
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response

# def chatbot_response(query):
#     # Configure sampling parameters for generation
#     sampling_params = SamplingParams(max_length=150, temperature=0.7, top_p=0.9)
    
#     # Generate a response from LLaMA using vLLM
#     response = llm.generate(query, sampling_params=sampling_params)
    
#     # Return the text of the response
#     return response[0].outputs[0].text
