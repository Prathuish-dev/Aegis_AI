import os
import json
import shutil
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

CATEGORY_MAPPING = {
    "data_drift_covariate_and_prior_probability.md": "data_issue",
    "llm_hallucinations_and_api_issues.md": "prompt_issue",
    "model_performance_and_accuracy_drops.md": "model_issue",
    "prompt_engineering_guidelines.md": "prompt_issue",
    "system_issues_oom_timeouts_rate_limits.md": "system_issue"
}

class KnowledgeBaseIngester:
    def __init__(
        self,
        embedding_function,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "aegis_knowledge"
    ):
        """
        Initializes the KnowledgeBaseIngester with an injected embedding function.
        
        Args:
            embedding_function: The LangChain-compatible embedding model instance.
            persist_directory: Path where ChromaDB sqlite files will be saved.
            collection_name: The name of the collection in ChromaDB.
        """
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Ensure the directory exists
        Path(self.persist_directory).parent.mkdir(parents=True, exist_ok=True)
        
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory
        )

    def _determine_category(self, filename: str) -> str:
        """Determines the category of a markdown guide based on its filename."""
        if filename in CATEGORY_MAPPING:
            return CATEGORY_MAPPING[filename]
        
        lower_name = filename.lower()
        if "drift" in lower_name:
            return "data_issue"
        elif "hallucination" in lower_name or "prompt" in lower_name:
            return "prompt_issue"
        elif "accuracy" in lower_name or "performance" in lower_name or "model" in lower_name:
            return "model_issue"
        elif "system" in lower_name or "timeout" in lower_name or "oom" in lower_name or "rate" in lower_name:
            return "system_issue"
        return "model_issue"

    def ingest_markdown_guides(self, directory_path: str = "data/knowledge_base") -> int:
        """
        Parses all markdown files directly in the given directory (ignoring subdirectories),
        chunks them, and adds them to ChromaDB.
        
        Returns:
            The number of chunks ingested.
        """
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} does not exist. Skipping markdown ingestion.")
            return 0
            
        documents = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n## ", "\n### ", "\n\n", "\n", " "]
        )

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) and filename.endswith(".md"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    category = self._determine_category(filename)
                    # Use relative path or basename as source
                    source = os.path.relpath(file_path).replace("\\", "/")
                    
                    chunks = splitter.split_text(content)
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "source": source,
                                "category": category,
                                "type": "guide",
                                "chunk_index": i
                            }
                        )
                        documents.append(doc)
                except Exception as e:
                    print(f"Error reading markdown guide {filename}: {e}")

        if documents:
            self.vector_store.add_documents(documents)
            self._persist()
            print(f"Ingested {len(documents)} markdown chunks into ChromaDB.")
            return len(documents)
        return 0

    def ingest_past_incidents(self, directory_path: str = "data/knowledge_base/past_incidents") -> int:
        """
        Parses all past incidents in JSON format from the given directory,
        structures them into a standard markdown text format, and adds them to ChromaDB.
        
        Returns:
            The number of incident records ingested.
        """
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} does not exist. Skipping past incidents ingestion.")
            return 0
            
        documents = []
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) and filename.endswith(".json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        incident = json.load(f)
                    
                    # Validate required keys
                    required_keys = ["incident_id", "failure_type", "root_cause", "fix_applied", "outcome"]
                    for key in required_keys:
                        if key not in incident:
                            raise ValueError(f"Missing required key '{key}' in incident {filename}")
                    
                    incident_id = incident["incident_id"]
                    failure_type = incident["failure_type"]
                    root_cause = incident["root_cause"]
                    fix_applied = incident["fix_applied"]
                    outcome = incident["outcome"]
                    tags = incident.get("tags", [])
                    tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
                    
                    # Format incident details as markdown content for semantically meaningful embeddings
                    page_content = (
                        f"# Incident {incident_id}\n"
                        f"- **Failure Type**: {failure_type}\n"
                        f"- **Root Cause**: {root_cause}\n"
                        f"- **Fix Applied**: {fix_applied}\n"
                        f"- **Outcome**: {outcome}\n"
                        f"- **Tags**: {tags_str}\n"
                    )
                    
                    source = os.path.relpath(file_path).replace("\\", "/")
                    
                    doc = Document(
                        page_content=page_content,
                        metadata={
                            "source": source,
                            "category": failure_type,
                            "type": "incident",
                            "incident_id": incident_id,
                            "tags": tags_str
                        }
                    )
                    documents.append(doc)
                except Exception as e:
                    print(f"Error reading past incident {filename}: {e}")

        if documents:
            self.vector_store.add_documents(documents)
            self._persist()
            print(f"Ingested {len(documents)} incident records into ChromaDB.")
            return len(documents)
        return 0

    def reset_vector_store(self) -> None:
        """Purges and resets the vector store database."""
        try:
            self.vector_store.delete_collection()
        except Exception as e:
            print(f"Note: delete_collection failed (likely collection didn't exist yet): {e}")
            
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                print(f"Removed persist directory {self.persist_directory} from disk.")
            except Exception as e:
                print(f"Error removing persist directory {self.persist_directory}: {e}")
                
        # Re-initialize ChromaDB
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory
        )

    def _persist(self) -> None:
        """Utility to persist vector store if the method exists in the wrapper version."""
        if hasattr(self.vector_store, "persist"):
            try:
                self.vector_store.persist()
            except Exception:
                pass
