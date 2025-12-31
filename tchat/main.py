from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Kamu adalah asisten yang membantu."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{content}")
])

chain = prompt | llm

history = []  # <-- memory manual (modern way)

while True:
    content = input(">> ")
    if content.lower() in {"exit", "quit"}:
        break

    history.append(HumanMessage(content=content))

    response = chain.invoke({
        "content": content,
        "history": history
    })

    print(response.content)

    history.append(AIMessage(content=response.content))
