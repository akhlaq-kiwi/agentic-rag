# src/tasks/rag_tasks.py
from crewai import Task, Crew
from crewai.process import Process


def create_rag_crew(retriever_agent, rag_agent, llm_agent):
    retriever_task = Task(
        description="Retrieve relevant chunks from pgvector for the query: {query}",
        agent=retriever_agent,
        expected_output="A list of relevant text chunks with metadata",
    )

    rag_task = Task(
        description="Take the retrieved chunks and format them into a structured context for answering the query",
        agent=rag_agent,
        expected_output="A formatted context string including metadata and relevant text",
        context=[retriever_task]
    )

    llm_task = Task(
        description="Answer the user query using the provided context. Be comprehensive and accurate.",
        agent=llm_agent,
        expected_output="A final human-readable answer to the query",
        context=[retriever_task, rag_task]
    )

    crew = Crew(
        agents=[retriever_agent, rag_agent, llm_agent],
        tasks=[retriever_task, rag_task, llm_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew
