from src.rag.crew import run_rag_query
from src.prompts.prompt_manager import get_prompt

# # Test with optimized RAG crew
user_message = "List out 10 core principles established by the Abu Dhabi Procurement Standards?"

print("🚀 Testing Optimized RAG Crew...")
print(f"Query: {user_message}")
print("="*80)

result = run_rag_query(user_message)
print(result.raw)

# prompt = get_prompt("insight_synthesizer_backstory")
# print(prompt.format().messages[0]['content'])
