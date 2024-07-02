import os
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate


# Try a Simple Example
os.environ['GOOGLE_API_KEY'] = "AIzaSyBTm4axDgCGC3Vc0ps8EMMnm7MDE0ZbjAM"
genai.configure(api_key = os.environ['GOOGLE_API_KEY'])
llm = ChatGoogleGenerativeAI(model="gemini-pro")
response = llm.invoke("Explain Quantum Computing in 50 words?")
print(response.content)


#Try With Prompt Template
chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "return the sentiment for the given text in json fromat the sentiment value can be 'nagative','positive'"
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)

chat_message =  chat_template.format_messages(text="i don't like weather today.")
llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3, convert_system_message_to_human=True) # set the convert_system_message_to_human to true

llm.invoke(chat_message)

parser = JsonOutputParser()

chain =  llm | parser

sentiment = chain.invoke(chat_message)

print(sentiment)


# Creating a Conversational bot with Gemini Pro and Langchain
PROMPT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI Assistant:
"""

PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=PROMPT_TEMPLATE
)
conversation = ConversationChain(
    llm=llm,
    prompt=PROMPT,
    memory=ConversationBufferMemory(ai_prefix="AI Assistant")
)
response = conversation.predict(input="who is Gojo?")
print(response)



