from src.rag.tools.rag_tools import pg_retriever_tool

user_message = "What should An employee with a disability shall be granted a fully-paid leave for a (5) working days per year at most?"

# rag_crew = create_rag_crew(user_message)
# result = rag_crew.kickoff()

data = pg_retriever_tool.run(user_message)
print(data)
