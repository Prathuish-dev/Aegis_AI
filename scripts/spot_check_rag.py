import os
import sys
from pathlib import Path
# No tabulate dependency

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from langchain_community.embeddings import HuggingFaceEmbeddings
from src.rag.retrieval import AegisRAG

def main():
    print("Loading embedding function...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    print("Initializing AegisRAG retriever...")
    rag = AegisRAG(
        embedding_function=embeddings,
        persist_directory="data/chroma_db"
    )
    
    # 5 Golden Test Cases covering each category
    golden_tests = [
        {
            "category": "data_issue",
            "query": "feature distribution shift and covariate drift",
            "expected_doc": "data_drift_covariate_and_prior_probability.md"
        },
        {
            "category": "prompt_issue",
            "query": "LLM hallucination, API rate limit, prompt guidelines",
            "expected_doc": "llm_hallucinations_and_api_issues.md"
        },
        {
            "category": "model_issue",
            "query": "sudden drop in model accuracy or F1 score",
            "expected_doc": "model_performance_and_accuracy_drops.md"
        },
        {
            "category": "system_issue",
            "query": "out of memory CUDA timeouts connection error",
            "expected_doc": "system_issues_oom_timeouts_rate_limits.md"
        },
        {
            "category": "incident_issue",
            "query": "INC-2026-004 memory leak in embedding service",
            "expected_doc": "incident_004.json"
        }
    ]
    
    results_table = []
    all_passed = True
    
    print("\nExecuting RAG Spot-Checks...")
    for test in golden_tests:
        query = test["query"]
        expected = test["expected_doc"]
        
        # We retrieve top 3 docs
        docs = rag.retrieve(query, k=3)
        retrieved_sources = [os.path.basename(doc.metadata.get("source", "")) for doc in docs]
        
        passed = expected in retrieved_sources
        if not passed:
            all_passed = False
            
        results_table.append([
            test["category"],
            query[:40] + "...",
            expected,
            ", ".join(retrieved_sources),
            "PASS" if passed else "FAIL"
        ])
        
    print("\n" + "="*80)
    print(f"{'Category':<15} | {'Expected Doc':<45} | {'Status'}")
    print("="*80)
    for row in results_table:
        category, _, expected, retrieved, status = row
        print(f"{category:<15} | {expected:<45} | {status}")
        print(f"  Retrieved: {retrieved}")
        print("-"*80)
    
    if all_passed:
        print("\nRAG Spot-Check PASSED! All golden documents retrieved in top 3.")
        sys.exit(0)
    else:
        print("\nRAG Spot-Check FAILED! Some golden documents were missing from top 3.")
        sys.exit(1)

if __name__ == "__main__":
    main()
