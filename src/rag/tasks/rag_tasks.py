# src/tasks/rag_tasks.py - OPTIMIZED VERSION
from crewai import Task, Crew, Process

def create_rag_crew(smart_retriever, answer_generator):
    """Create optimized RAG crew - reduced from 4 to 2 tasks for speed."""
    
    # TASK 1: Smart Retrieval (handles routing + retrieval in one step)
    retrieval_task = Task(
        description="""Handle the user query efficiently: {query}

        DECISION LOGIC:
        1. If query is a simple greeting (hi, hello, good morning, how are you):
           - Respond directly with a friendly greeting
           - DO NOT use pg_retriever_tool
           - Example: "Hello! I'm here to help you with document questions."

        2. If query asks for specific information (policies, procedures, facts):
           - Use pg_retriever_tool to search for relevant document chunks
           - Return the retrieved information for the next agent to process

        3. If query is unclear or too vague:
           - Ask for clarification without using tools
           - Example: "Could you please ask a more specific question about the documents?"

        Be efficient - handle simple cases directly, use tools only when needed.""",
        agent=smart_retriever,
        expected_output="Direct response for greetings/clarifications, or retrieved document chunks for factual queries"
    )

    # TASK 2: Answer Generation (streamlined)
    answer_task = Task(
        description="""Generate the final response based on the retrieval results: {query}

        PROCESSING LOGIC:
        1. If the previous task returned a direct response (greeting/clarification):
           - Use that response as-is
           
        2. If the previous task retrieved document chunks:
           - Extract relevant information from the chunks
           - Quote directly from documents: "According to [Document], page [X]: [information]"
           - Cite all sources used
           - If no relevant information found, say "I don't have information about that in the available documents."

        REQUIREMENTS:
        - Only use information explicitly stated in retrieved documents
        - Never add external knowledge or make assumptions
        - Keep responses concise and factual
        - Always provide source citations for document-based answers""",
        agent=answer_generator,
        expected_output="Final response - either direct greeting/clarification or document-based answer with citations",
        context=[retrieval_task]
    )

    # Create optimized crew with minimal overhead
    crew = Crew(
        agents=[smart_retriever, answer_generator],
        tasks=[retrieval_task, answer_task],
        process=Process.sequential,
        verbose=False,  # Reduced verbosity for speed
        memory=False,   # Disable memory for faster processing
        max_iter=1      # Prevent unnecessary iterations
    )

    return crew
