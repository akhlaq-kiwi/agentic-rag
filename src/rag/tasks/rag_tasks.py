# src/tasks/rag_tasks.py - OPTIMIZED VERSION
from crewai import Task, Crew, Process
from ..agents.rag_agent import document_researcher, insight_synthesizer, query_analyzer

def create_rag_crew(query: str):
    parts = query.split("<-->")
    query = parts[0]
    context = parts[1] if len(parts) > 1 else ""
    
    """
    Creates and configures a two-agent RAG crew to process a query.
    - The Document Researcher finds relevant information.
    - The Insight Synthesizer formulates the final answer based on the retrieved context.
    """

    query_analyzer_task = Task(
        name="Query Analyzer Task",
        description=f"Analyze the user's query to determine the best approach for information retrieval for the query: '{query}'.",
        expected_output="A string indicating the type of query: 'greeting', 'chat_history', or 'question'.",
        agent=query_analyzer
    )

    # Task for the Document Researcher agent
    # This task focuses exclusively on using the tool to find information.
    research_task = Task(
        name="Document Researcher Task",
        description=(
            f"Given the user query: '{query}', decide the appropriate response "
            "based on the analysis from the 'Query Analyzer Task'."
        ),
        expected_output=f"""
        - If Query Analyzer Task → intent = 'greeting':
           Return only a concise and friendly greeting message.
           Do not invoke any tools or search documents.

        - If Query Analyzer Task → intent = 'chat_history':
           Retrieve and return the most relevant answer from the "{context}".
           Do not perform document search or other tasks.

        - If Query Analyzer Task → intent = 'question':
           Return a structured response that includes:
              1. A block of text containing the most relevant chunks from policy and standards documents.
              2. For each chunk, include its source file name (and page number if available).
              3. Ensure the text is directly extracted, not paraphrased, so it can be cited as evidence.
        """,
        agent=document_researcher
    )
    
    # Task for the Insight Synthesizer agent
    # This task takes the context from the first task and focuses on crafting the answer.
    synthesis_task = Task(
        name="Insight Synthesizer Task",
        description=f"Analyze the provided document context from 'Document Researcher Task' and formulate a comprehensive and accurate answer to the user's original question: '{query}'.",
        expected_output="""A professional, well-structured response that directly answers the user's question with proper source citations:

        RESPONSE REQUIREMENTS:
        - Start with a clear, direct answer to the question
        - Provide supporting details and explanations as relevant
        - MANDATORY: Include specific source citations throughout the response
        - Use natural language flow with integrated citations
        - Adapt structure to content complexity (simple answers for simple questions, detailed for complex ones)
        - Use appropriate formatting (bullet points, numbering, or paragraphs)
        - Maintain professional yet conversational tone

        CITATION FORMAT:
        - Integrate citations naturally: "According to the Procurement Manual (Page 6)..."
        - Reference specific documents and page numbers from the research context
        - End with a "Sources:" section listing all referenced documents with page numbers
        - Example: "Sources: Procurement Manual (Business Process).PDF (Page 6, 7), Abu Dhabi Procurement Standards.PDF (Page 12)"

        QUALITY STANDARDS:
        - Every factual claim must be traceable to the provided context
        - Preserve accuracy by staying within the retrieved information
        - If information is insufficient, clearly state what's missing
        - Include precise figures, timeframes, and regulatory references where available

        The response should demonstrate clear traceability between claims and sources while maintaining readability.""",
        agent=insight_synthesizer,
        context=[research_task]
    )

    # Create the crew with a sequential process
    rag_crew = Crew(
        agents=[query_analyzer, document_researcher, insight_synthesizer],
        tasks=[query_analyzer_task, research_task, synthesis_task],
        process=Process.sequential, # The tasks will be executed one after the other
        verbose=True
    )

    return rag_crew