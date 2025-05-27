import os
from dotenv import load_dotenv
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Load environment variables from .env file
load_dotenv()

# Ensure GOOGLE_API_KEY is set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please create a .env file.")

# 1. Initialize the Language Model (LLM)
# Using Gemini 1.5 Flash for faster responses. You can choose other models like 'gemini-pro'.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# 2. Define the Prompt Template
# The prompt template guides the LLM on how to behave.
# We include {chat_history} for conversational memory.
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer user questions concisely and politely."),
    ("human", "{question}"),
])

# 3. Set up Conversational Memory
# This allows the chatbot to remember previous turns in the conversation.
memory = ConversationBufferMemory(memory_key="chat_history")

# 4. Create an LLM Chain with Memory
# This chain combines the prompt, LLM, and memory for a conversational flow.
# We explicitly pass the memory to the chain.
conversation_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    verbose=True, # Set to True to see the intermediate steps
    memory=memory
)

# 5. Chatbot Interaction Loop
def run_chatbot():
    print("Welcome to the LangChain Chatbot! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        # Invoke the chain with the user's question
        # The 'question' key in the prompt_template is populated by user_input.
        # The 'chat_history' is automatically managed by the ConversationBufferMemory.
        response = conversation_chain.invoke({"question": user_input})
        
        # LangChain's LLMChain returns a dictionary, the output is in the 'text' key.
        print(f"Bot: {response['text']}")

if __name__ == "__main__":
    run_chatbot()