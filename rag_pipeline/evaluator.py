import typing as T
import pandas as pd
from ragas import evaluate
from ragas.metrics import Faithfulness, FactualCorrectness, LLMContextRecall
from datasets import Dataset
from langchain_core.language_models.chat_models import BaseChatModel


class RAGASEvaluator:
    """Handles the evaluation of RAG pipeline results using RAGAS metrics"""
    def __init__(self, evaluator_llm: BaseChatModel) -> None:
        self.evaluator_llm = evaluator_llm
        # -- Using relevant RAG metrics: Change as required -- #
        self.metrics = [Faithfulness(), FactualCorrectness(), LLMContextRecall()]
    
    def evaluate_results(self, rag_results: T.List[T.Dict[str, str]]) -> pd.DataFrame:
        """
        Performs RAGAS evaluation on the collected RAG results.

        Args:
            rag_results: A list of dictionaries containing 'question', 'answer', 'contexts', and 'ground_truth'.

        Returns:
            A pandas DataFrame with the RAGAS evaluation scores.
        """
        data = {
            "question": [r["question"] for r in rag_results],
            "answer": [r["answer"] for r in rag_results],
            "contexts": [r["contexts"] for r in rag_results],
            "ground_truth": [r["ground_truth"] for r in rag_results],
        }
        dataset = Dataset.from_dict(data)

        print("Performing RAGAS evaluation...")
        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.evaluator_llm
        )
        return result.to_pandas()