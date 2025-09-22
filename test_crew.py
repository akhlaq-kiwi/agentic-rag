from src.rag.crew import run_rag_query

# Test with optimized RAG crew
user_message = "Explain 2.10.2 'SAP Ariba - Core Administration' Overview?"

print("🚀 Testing Optimized RAG Crew...")
print(f"Query: {user_message}")
print("="*80)

result = run_rag_query(user_message)
print(result)
