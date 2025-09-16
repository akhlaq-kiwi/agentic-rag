from crew import create_rag_crew
from tools import document_retrieval_tool

user_message = "Article ( 2 ) Scope of Application"

rag_crew = create_rag_crew(user_message)
result = rag_crew.kickoff()

# data = document_retrieval_tool.run(user_message)
# print(data)
