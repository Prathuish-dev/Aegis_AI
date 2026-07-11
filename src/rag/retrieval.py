import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma

class AegisRAG:
    def __init__(
        self,
        embedding_function,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "aegis_knowledge"
    ):
        """
        Initializes the AegisRAG retriever with an injected embedding function.
        
        Args:
            embedding_function: The LangChain-compatible embedding model instance.
            persist_directory: Path where ChromaDB sqlite files are saved.
            collection_name: The name of the collection in ChromaDB.
        """
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize the Chroma store (read-only mode if DB exists)
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory
        )

    def retrieve(self, query: str, category: str = None, k: int = 5) -> list[Document]:
        """
        Retrieves relevant documents using Maximal Marginal Relevance (MMR) search.
        
        Args:
            query: The search query string.
            category: Optional category metadata filter (must match FailureCategory string values).
            k: The number of documents to return (defaults to 5).
            
        Returns:
            A list of retrieved LangChain Document objects.
        """
        search_kwargs = {
            "k": k,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
        
        # Apply category filter if provided
        if category:
            search_kwargs["filter"] = {"category": category}
            
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs
        )
        
        try:
            return retriever.invoke(query)
        except Exception as e:
            print(f"Error executing MMR retrieval: {e}")
            # Fallback to standard similarity search if MMR fails
            try:
                filter_dict = {"category": category} if category else None
                return self.vector_store.similarity_search(query, k=k, filter=filter_dict)
            except Exception as e2:
                print(f"Error executing fallback similarity search: {e2}")
                return []
