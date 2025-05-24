import os
import json

from rag_pipeline.model_provider import LLMProvider, EmbeddingProvider
from rag_pipeline.dataloader import DollyDataLoader
from rag_pipeline.pipeline import RAGPipeline
from rag_pipeline.evaluator import RAGASEvaluator

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
EVALUATOR_LLM_NAME = os.getenv("EVALUATOR_LLM_NAME", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")

DOCUMENTS_SAMPLE_SIZE = int(os.getenv("DOCUMENTS_SAMPLE_SIZE", 100))
QUERY_SAMPLE_SIZE = int(os.getenv("QUERY_SAMPLE_SIZE", 5))
print(f"EMBEDDING_MODEL_NAME: {EMBEDDING_MODEL_NAME}")
print(f"LLM_MODEL_NAME: {LLM_MODEL_NAME}")
print(f"EVALUATOR_LLM_NAME: {EVALUATOR_LLM_NAME}")
print(f"DOCUMENTS_SAMPLE_SIZE: {DOCUMENTS_SAMPLE_SIZE}")
print(f"QUERY_SAMPLE_SIZE: {QUERY_SAMPLE_SIZE}")

def run_evaluation():
    dataloader = DollyDataLoader(sample_size=DOCUMENTS_SAMPLE_SIZE)
    documents_for_vectorstore, hf_dataset_for_queries = dataloader.load_data()
    if len(hf_dataset_for_queries) == 0:
        print("No samples found matching the criteria. Please check dataset name, category, and sample size.")
        exit(1)

    print(f"Number of documents loaded for vector store: {len(documents_for_vectorstore)}")
    print(f"Number of evaluation samples from HF dataset (before query sampling): {len(hf_dataset_for_queries)}")

    embedding_provider = EmbeddingProvider(model_name=EMBEDDING_MODEL_NAME).get_embedding_provider()
    llm_provider = LLMProvider(model_name=LLM_MODEL_NAME).get_llm()
    evaluator_llm_provider = LLMProvider(model_name=EVALUATOR_LLM_NAME).get_llm()

    rag_pipeline = RAGPipeline(llm_provider, embedding_provider)
    rag_pipeline.build_vector_store(documents_for_vectorstore)
    rag_pipeline.setup_qa_chain()

    print("Running queries from Dolly dataset and collecting data for RAGAS...")
    rag_results = rag_pipeline.run_queries(hf_dataset_for_queries, QUERY_SAMPLE_SIZE)

    with open("rag_results.json", "w") as _f:
        json.dump(rag_results, _f, indent=4)
    print("RAG query results saved to rag_results.json")

    ragas_evaluator = RAGASEvaluator(evaluator_llm_provider)
    evaluation_df = ragas_evaluator.evaluate_results(rag_results)


    faithfulness_score = evaluation_df["faithfulness"].mean() # Ragas returns a score for each sample, use mean for overall
    context_recall_score = evaluation_df["context_recall"].mean()
    factual_correctness_score = evaluation_df["factual_correctness(mode=f1)"].mean()

    # print("\n--- RAGAS Evaluation Results ---")
    print(f"Average Faithfulness Score: {faithfulness_score:.2f}")
    print(f"Average Context Recall Score: {context_recall_score:.2f}")
    print(f"Average Factual Correctness Score: {factual_correctness_score:.2f}")

    evaluation_df.to_csv("ragas_results.csv", index=False)
    # print("\nEvaluation results saved to ragas_results.csv")



if __name__ == "__main__":
    run_evaluation()