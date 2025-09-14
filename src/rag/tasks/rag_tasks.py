# src/tasks/rag_tasks.py
from crewai import Task, Crew


def create_rag_crew(retriever_agent, rag_agent, llm_agent):
    retriever_task = Task(
        description="Retrieve relevant chunks from pgvector for the query",
        agent=retriever_agent,
        expected_output="A list of relevant text chunks with metadata",
    )

    rag_task = Task(
        description="Take retrieved chunks and format them into a structured context",
        agent=rag_agent,
        expected_output="A formatted context string including metadata and relevant text",
    )

    llm_task = Task(
        description="Answer the user query using retrieved context",
        agent=llm_agent,
        expected_output="A final human-readable answer",
    )

    crew = Crew(
        agents=[retriever_agent, rag_agent, llm_agent],
        tasks=[retriever_task, rag_task, llm_task],
        verbose=True,
    )

    return crew
