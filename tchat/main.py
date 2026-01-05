# main.py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import FileChatMessageHistory

def main():
    # Load environment variables (OPENAI_API_KEY, etc.)
    load_dotenv()

    # LLM
    llm = ChatOpenAI(model="gpt-5.2", verbose=True)

    # Persistent chat history (saved to disk)
    history = FileChatMessageHistory(file_path="chat_history.json")

    # Prompt template includes:
    # - system instruction
    # - conversation history
    # - latest user message
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Kamu adalah asisten yang membantu."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{content}")
    ])

    # Chain: prompt -> llm
    chain = prompt | llm

    print("Chat started. Type 'exit' or 'quit' to stop.")
    while True:
        content = input(">> ").strip()
        if not content:
            continue
        if content.lower() in {"exit", "quit"}:
            break

        # 1) Save user message to persistent history
        history.add_user_message(content)

        # 2) Invoke model with full history
        response = chain.invoke({
            "content": content,
            "history": history.messages
        })

        # 3) Print assistant response
        print(response.content)

        # 4) Save assistant response to persistent history
        history.add_ai_message(response.content)

    print("Bye!")

if __name__ == "__main__":
    main()
