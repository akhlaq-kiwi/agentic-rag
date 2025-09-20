# src/tasks/rag_tasks.py
from crewai import Task, Crew
from crewai import Crew, Task, Process

def create_rag_crew(retriever_agent, answer_agent):
    retriever_task = Task(
        description="Retrieve relevant document chunks for the query: {query}",
        agent=retriever_agent,
        expected_output="Retrieved document chunks with source information and metadata",
    )

    answer_task = Task(
        description="""Based on the retrieved document chunks, provide a comprehensive answer to the user query: {query}
        
        Instructions:
        - Use only the information from the retrieved chunks
        - Provide a clear, well-structured answer
        - Include relevant details and examples from the documents
        - If multiple sources are relevant, synthesize the information
        - If the retrieved context is insufficient, clearly state what information is missing
        - Cite sources when appropriate (mention document names or sections)""",
        agent=answer_agent,
        expected_output="A comprehensive, accurate answer to the user's query based on the retrieved context",
        context=[retriever_task]
    )

    crew = Crew(
        agents=[retriever_agent, answer_agent],
        tasks=[retriever_task, answer_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew
