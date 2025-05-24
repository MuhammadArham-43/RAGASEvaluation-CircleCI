import typing as T
import os
from datasets import load_dataset, Dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

class DollyDataLoader:
    """Loads and preprocesses the Dolly dataset for RAG evaluation"""
    def __init__(
        self,
        dataset_path: str = "databricks/databricks-dolly-15k",
        category: str = "closed_qa",
        split: str = "train",
        sample_size: T.Union[int, None] = None
    ) -> None:
        self.dataset_path = dataset_path
        self.category = category
        self.sample_size = sample_size
        self.split = split
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    
    def load_data(self) -> T.Tuple[T.List[Document], Dataset]:
        """
        Loads the Dolly dataset, filters by closed_qa category, samples, and prepares documents for the vector store.

        Returns:
            A tuple containing:
            - List[Document]: LangChain Document objects for the vector store.
            - Dataset: Sampled Hugging Face dataset for RAG queries and ground truths.
        """

        dataset = load_dataset(self.dataset_path, split=self.split)
        dataset = dataset.filter(
            lambda x: x['category'] == self.category and x['context'] is not None and x['context'].strip() != "",
            num_proc=os.cpu_count()
        )
        if self.sample_size:
            dataset = dataset.shuffle(seed=42).select(
                range(min(self.sample_size, len(dataset)))
            )
        
        documents_content = [item["context"] for item in dataset]
        langchain_documents = [Document(page_content=content) for content in documents_content]
        return self.text_splitter.split_documents(langchain_documents), dataset