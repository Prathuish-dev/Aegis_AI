import os
import argparse
import sys
from pathlib import Path

# Add project root to sys.path to enable importing src modules
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from src.rag.ingestion import KnowledgeBaseIngester
from langchain_community.embeddings import HuggingFaceEmbeddings

def main():
    parser = argparse.ArgumentParser(description="Rebuild Aegis AI RAG Knowledge Base in ChromaDB")
    parser.add_argument(
        "--path",
        type=str,
        default="data/knowledge_base",
        help="Path to the knowledge base root directory (contains markdown guides and past_incidents/)"
    )
    parser.add_argument(
        "--persist_dir",
        type=str,
        default="data/chroma_db",
        help="Directory where the Chroma DB is stored"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Purge/reset the Chroma DB before re-indexing"
    )
    args = parser.parse_args()

    print("Initializing embedding function (BAAI/bge-small-en-v1.5)...")
    try:
        # Load local HuggingFace embedding model as recommended in guidelines
        embedding_function = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        sys.exit(1)

    print(f"Initializing KnowledgeBaseIngester with persist_dir: {args.persist_dir}")
    ingester = KnowledgeBaseIngester(
        embedding_function=embedding_function,
        persist_directory=args.persist_dir
    )

    if args.reset:
        print("Purging existing vector store...")
        ingester.reset_vector_store()
        print("Vector store reset complete.")

    # Ingest markdown guides
    print(f"Ingesting markdown guides from: {args.path}")
    guides_count = ingester.ingest_markdown_guides(directory_path=args.path)

    # Ingest past incidents
    incidents_path = os.path.join(args.path, "past_incidents")
    print(f"Ingesting past incidents from: {incidents_path}")
    incidents_count = ingester.ingest_past_incidents(directory_path=incidents_path)

    print("\n--- Rebuild Complete ---")
    print(f"Total markdown chunks ingested: {guides_count}")
    print(f"Total past incidents ingested: {incidents_count}")
    print(f"ChromaDB persist directory: {args.persist_dir}")

if __name__ == "__main__":
    main()
