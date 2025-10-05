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
            f"For the query: '{query}', call pg_retriever_tool ONCE and return its complete output. "
            "Do not call the tool multiple times. Do not modify the output."
        ),
        expected_output=(
            "The complete, unmodified output from pg_retriever_tool containing retrieved document chunks with sources. "
            "If this is a greeting, return a friendly message WITHOUT calling the tool. "
            "Call the tool ONCE only - do not retry regardless of results."
        ),
        agent=document_researcher
    )
    
    # Task for the Insight Synthesizer agent
    # This task takes the context from the first task and focuses on crafting the answer.
    synthesis_task = Task(
        name="Insight Synthesizer Task",
        description=(
            f"Using ONLY the output from 'Document Researcher Task', create a response to: '{query}'. "
            f"CRITICAL: Use ZERO external knowledge. If the research output doesn't contain the answer, say so explicitly. "
            f"Do not use your training data or general knowledge under any circumstances."
        ),
        expected_output="""A response based EXCLUSIVELY on the Document Researcher's output with proper source citations:

        RESPONSE REQUIREMENTS:
        - Start with a clear, direct answer to the question
        - Provide supporting details and explanations as relevant
        - MANDATORY: Include specific source citations throughout the response
        - Use natural language flow with integrated citations
        - Adapt structure to content complexity (simple answers for simple questions, detailed for complex ones)
        - Use appropriate formatting (bullet points, numbering, or paragraphs)
        - Maintain professional yet conversational tone

        CITATION FORMAT - CRITICAL:
        - The Document Researcher's output contains chunks with "Source: [Filename] (Page X)" lines
        - You MUST copy the EXACT page numbers from these "Source:" lines - DO NOT modify them
        - DO NOT use section numbers (like 2.9.3) as page numbers - ONLY use actual page numbers from "Source:" lines
        - DO NOT infer, guess, or create page numbers - ONLY copy what appears after "Page" in the source line
        - End with a "Sources:" section listing ALL referenced documents with their EXACT page numbers
        - Example: If source says "Source: Procurement Manual (Business Process).PDF (Page 83)", you write "Sources: Procurement Manual (Business Process).PDF (Page 83)"
        - WRONG: Using section numbers like "Page 2.9.3" when the source says "Page 83"
        - RIGHT: Copying exact page number "Page 83" from the source line
        - MANDATORY: Every fact must be traceable to a specific "Source:" line with EXACT page number from the research output

        CRITICAL - ZERO EXTERNAL KNOWLEDGE:
        - Use ONLY information from the Document Researcher's output
        - Do NOT use your training data, general knowledge, or make assumptions
        - If the research output lacks information, explicitly state: "The provided documents do not contain information about [topic]"
        - Every factual claim must be directly from the research context with source citation
        - If you cannot answer from the research context alone, say so clearly

        QUALITY STANDARDS:
        - Every statement must be traceable to a specific document and page number
        - Preserve accuracy by staying strictly within the retrieved information
        - Include precise figures, timeframes, and references ONLY if they appear in the research output
        - If information is insufficient, clearly state what's missing - do not fill gaps with general knowledge

        IMPORTANT RESTRICTIONS:
        - Do NOT generate followup questions or suggested questions
        - Do NOT add sections like "Related questions:", "You might also ask:", or "Next steps:"
        - Do NOT use external knowledge or training data
        - End your response with the Sources section - nothing after that

        The response must be 100% grounded in the Document Researcher's output with clear source attribution.""",
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