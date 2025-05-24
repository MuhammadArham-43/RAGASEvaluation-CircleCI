import typing as T
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.docstore.document import Document
from datasets import Dataset

class RAGPipeline:
    def __init__(self, llm: BaseChatModel, embedding_provider: Embeddings) -> None:
        self.llm = llm
        self.embedding_provider = embedding_provider
        self.vectorstore: FAISS = None
        self.qa_chain: RetrievalQA = None
    
    def build_vector_store(self, documents: T.List[Document]) -> None:
        print("Embedding documents with FAISS vectorstore")
        self.vectorstore = FAISS.from_documents(documents, self.embedding_provider)
    
    def setup_qa_chain(self) -> None:
        if not self.vectorstore:
            self.build_vector_store()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever()
        )
    
    def run_queries(
        self,
        hf_dataset_sample: Dataset,
        query_sample_size: int = 30
    ) -> T.List[T.Dict[str, str]]:
        """
        Runs RAG queries against the pipeline and collects results for RAGAS evaluation.

        Args:
            hf_dataset_sample: The sampled Hugging Face dataset containing questions and ground truths.
            query_sample_size: The number of queries to run from the sampled dataset.

        Returns:
            A list of dictionaries, each containing 'question', 'answer', 'contexts', and 'ground_truth'.
        """
        if not self.qa_chain:
            self.setup_qa_chain()
        
        sampled_queries_dataset = hf_dataset_sample.shuffle(seed=42).select(
            range(min(query_sample_size, len(hf_dataset_sample)))
        )
        results = []
        for item in tqdm(sampled_queries_dataset, desc="Generating responses for queries"):
            query = item["instruction"]
            ground_truth = item["response"]

            # Generates response for a query
            response = self.qa_chain.invoke({"query": query})   
            answer = response["result"]

            # Only fetches the relevant documents from knowlege base used as context for the response
            retrieved_docs = self.qa_chain.retriever.invoke(query)  
            contexts = [doc.page_content for doc in retrieved_docs]

            results.append({
                "question": query,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth
            })
        return results