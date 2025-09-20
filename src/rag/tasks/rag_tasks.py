# src/tasks/rag_tasks.py
from crewai import Task, Crew
from crewai import Crew, Task, Process

def create_rag_crew(query_router, greeting_handler, document_retriever, answer_agent):
    routing_task = Task(
        description="""Analyze the user query: {query}
        
        Classify the query and respond with one of these formats:
        - For greetings (hello, hi, good morning, etc.): "GREETING: [friendly response]"
        - For document questions: "RETRIEVE: [original query]"
        - For unclear queries: "CLARIFY: Please ask a specific question about the documents."
        """,
        agent=query_router,
        expected_output="Classification of the query with appropriate routing decision",
    )

    greeting_task = Task(
        description="""Handle greetings and clarifications based on the routing decision:

        - If routing decision starts with 'GREETING:', return the greeting message directly
        - If routing decision starts with 'CLARIFY:', return the clarification message directly  
        - If routing decision starts with 'RETRIEVE:', return "NEEDS_RETRIEVAL"
        
        You have no tools available - just process the routing decision appropriately.""",
        agent=greeting_handler,
        expected_output="Greeting response, clarification message, or 'NEEDS_RETRIEVAL' indicator",
        context=[routing_task]
    )

    retrieval_task = Task(
        description="""Only execute if the greeting handler returned 'NEEDS_RETRIEVAL':
        
        Use pg_retriever_tool to search for relevant document chunks for the query: {query}
        
        If greeting handler did NOT return 'NEEDS_RETRIEVAL', skip this task.""",
        agent=document_retriever,
        expected_output="Retrieved document chunks with source information and metadata, or empty if not needed",
        context=[routing_task, greeting_task]
    )

    answer_task = Task(
        description="""Generate the final response with ZERO creativity and STRICT document adherence:

        STEP 1: Examine the retrieved document chunks from the previous task
        STEP 2: Look for ANY information relevant to the query: {query}
        STEP 3: If relevant information exists, use it to answer (even if partial)

        RESPONSE TYPES:

        1. GREETING RESPONSES (routing starts with 'GREETING:'):
           - Return the exact greeting message from the routing decision
           - NO modifications or additions
           
        2. CLARIFICATION RESPONSES (routing starts with 'CLARIFY:'):
           - Return the exact clarification message from routing
           - NO modifications or additions
           
        3. DOCUMENT-BASED RESPONSES (routing starts with 'RETRIEVE:'):
           - FIRST: Examine the retrieved document chunks carefully
           - IF chunks contain ANY relevant information, use it to answer the question
           - Quote directly from documents when possible
           - NEVER add information not present in the retrieved chunks
           - NEVER use general knowledge, assumptions, or inferences
           - ONLY say "I don't have enough information" if chunks are completely empty or totally irrelevant
           - Always cite: "According to [Document Name], page [X]: [exact quote or information]"
           - If partial information is available, provide what you can find and cite sources

        CRITICAL REQUIREMENTS:
        - Temperature = 0 (no creativity)
        - Stick to facts explicitly stated in documents
        - Never expand beyond retrieved content
        - Never make logical connections not explicitly stated
        - Quote exact text when possible
        
        Query to answer: {query}
        """,
        agent=answer_agent,
        expected_output="Exact document-based response with zero creativity and strict source adherence",
        context=[routing_task, greeting_task, retrieval_task]
    )

    crew = Crew(
        agents=[query_router, greeting_handler, document_retriever, answer_agent],
        tasks=[routing_task, greeting_task, retrieval_task, answer_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew
