import os
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ['GOOGLE_API_KEY'] = ""
genai.configure(api_key = os.environ['GOOGLE_API_KEY'])

llm = ChatGoogleGenerativeAI(model="gemini-pro")
response = llm.invoke("Explain Quantum Computing in 50 words?")
print(response.content)