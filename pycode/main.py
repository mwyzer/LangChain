from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of alphabets")
parser.add_argument("--language", default="Python")
args = parser.parse_args()

load_dotenv()

llm = OpenAI(temperature=0.2)

code_prompt = PromptTemplate(
    template="Write a very short {language} program to {task}. Output code only.",
    input_variables=["language", "task"],
)

test_prompt = PromptTemplate(
    template=(
        "You are a QA engineer.\n"
        "Write a minimal test in {language} to verify the functionality of this code.\n"
        "Output code only.\n\n"
        "CODE:\n```{language}\n{code}\n```"
    ),
    input_variables=["language", "code"],
)

code_chain = code_prompt | llm
test_chain = test_prompt | llm

# Step 1: generate code
code = code_chain.invoke({
    "language": args.language,
    "task": args.task
})

# Step 2: generate test code
test_code = test_chain.invoke({
    "language": args.language,
    "code": code
})

print("=== GENERATED CODE ===")
print(code)
print("\n=== GENERATED TEST ===")
print(test_code)
print("\n=== END ===")