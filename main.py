import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_together import TogetherEmbeddings, ChatTogether

from langchain.chains.retrieval_qa.base import RetrievalQA
from ragas import evaluate
from ragas.metrics import Faithfulness, LLMContextRecall, FactualCorrectness
import pandas as pd
from datasets import load_dataset, Dataset
from langchain.docstore.document import Document



EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
LLM_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
EVALUATOR_LLM_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
HF_DATASET_NAME = "databricks/databricks-dolly-15k"
DOLLY_CATEGORY = "closed_qa"
DOCUEMNTS_SAMPLE_SIZE = 100    # Number of samples to use from the dataset for quick evaluation
QUERY_SAMPLE_SIZE = 10


def create_corpus_from_dolly_dataset(dataset_name, category, sample_size):
    dataset = load_dataset(dataset_name, split="train")
    filtered_dataset = dataset.filter(
        lambda x: x['category'] == category and x['context'] is not None and x['context'].strip() != "",
        num_proc=os.cpu_count()
    )

    sampled_dataset = filtered_dataset.shuffle(seed=42).select(range(min(sample_size, len(filtered_dataset))))
    documents_content = [item['context'] for item in sampled_dataset]
    langchain_documents = [Document(page_content=content) for content in documents_content]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(langchain_documents), sampled_dataset

def create_vector_store(documents, embedding_model_name):
    embedding_provider = TogetherEmbeddings(model=embedding_model_name)
    vectorstore = FAISS.from_documents(documents, embedding_provider)
    return vectorstore

def initialize_llm(model_name):
    llm = ChatTogether(model=model_name)
    return llm

def get_evaluator_llm(model_name):
    return ChatTogether(model=model_name)


def setup_rag_pipeline(vectorstore, llm):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    return qa_chain

def run_rag_queries(qa_chain, hf_dataset_sample, query_sample_size: int = 10):
    hf_dataset_sample = hf_dataset_sample.shuffle(seed=42).select(range(min(query_sample_size, len(hf_dataset_sample))))
    results = []
    for i, item in enumerate(hf_dataset_sample):
        query = item["instruction"]
        ground_truth = item["response"]
        print(f"Processing query {i+1}/{len(hf_dataset_sample)}: {query[:50]}...")
        response = qa_chain.invoke({"query": query})
        answer = response["result"]
        retrieved_docs = qa_chain.retriever.invoke(query)
        contexts = [doc.page_content for doc in retrieved_docs]

        results.append({
            "question": query,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth
        })
    
    return results

def perform_ragas_evaluation(rag_results):
    data = {
        "question": [r["question"] for r in rag_results],
        "answer": [r["answer"] for r in rag_results],
        "contexts": [r["contexts"] for r in rag_results],
        "ground_truth": [r["ground_truth"] for r in rag_results],
    }
    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), LLMContextRecall(), FactualCorrectness()],
        llm=get_evaluator_llm(EVALUATOR_LLM_NAME)
    )
    return result

if __name__ == "__main__":
    documents, hf_dataset_sample = create_corpus_from_dolly_dataset(
        HF_DATASET_NAME, DOLLY_CATEGORY, DOCUEMNTS_SAMPLE_SIZE
    )

    print(f"Number of documents loaded for vector store: {len(documents)}")
    print(f"Number of evaluation samples from HF dataset: {len(hf_dataset_sample)}")

    if len(hf_dataset_sample) == 0:
        print("No samples found matching the criteria. Please check dataset name, category, and sample size.")
        exit(1)


    print(f"Creating vector store with {EMBEDDING_MODEL_NAME} embeddings...")
    vectorstore = create_vector_store(documents, EMBEDDING_MODEL_NAME)

    print(f"Initializing LLM with {LLM_MODEL_NAME}...")
    llm = initialize_llm(LLM_MODEL_NAME)

    print("Setting up RAG pipeline...")
    qa_chain = setup_rag_pipeline(vectorstore, llm)

    print("Running queries from Dolly dataset and collecting data for RAGAS...")
    rag_results = run_rag_queries(qa_chain, hf_dataset_sample, QUERY_SAMPLE_SIZE)
    with open("rag_results.json", "w") as _f:
        json.dump(rag_results, _f, indent=4)
    print("Performing RAGAS evaluation...")
    evaluation_results = perform_ragas_evaluation(rag_results)

    print("\n--- RAGAS Evaluation Results ---")
    print(evaluation_results)