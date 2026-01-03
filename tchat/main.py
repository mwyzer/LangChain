from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Kamu adalah asisten yang membantu."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{content}")
])

chain = prompt | llm

# --- Memory Setup ---
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap the chain with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="content",
    history_messages_key="history",
)

# --- Chat Loop ---
session_id = "user_session_1"  # Static session ID for this script

while True:
    content = input(">> ")
    if content.lower() in {"exit", "quit"}:
        break

    # history is now managed automatically by the wrapper
    response = chain_with_history.invoke(
        {"content": content},
        config={"configurable": {"session_id": session_id}}
    )

    print(response.content)
