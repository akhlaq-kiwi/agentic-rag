from crewai.tools import BaseTool

class PgRetrieverTool(BaseTool):
    name: str = "pg_retriever"
    description: str = "Retrieve relevant chunks from pgvector using semantic similarity"

    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever

    def _run(self, query: str) -> str:
        results = self.retriever.retrieve(query)
        return "\n\n".join([r.node.get_content() for r in results])
