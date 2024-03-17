import dotenv

dotenv.load_dotenv()

from ragas import evaluate
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.docstore.document import Document
from ragas.llms import LangchainLLMWrapper, llm_factory
from ragas.metrics.base import MetricWithLLM
from ragas.testset.generator import TestDataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain.document_loaders.base import Document
from ragas.metrics.critique import harmfulness


def download_semantic_scholar_document(
    query: str = "Large Language Models", top_k: int = 10
) -> list[Document]:
    from llama_index import download_loader

    SemanticScholarReader = download_loader("SemanticScholarReader")
    loader = SemanticScholarReader()
    query_space = query
    llamaindex_documents = loader.load_data(
        query=query_space, full_text=True, limit=top_k
    )
    lc_documents = [doc.to_langchain_format() for doc in llamaindex_documents]
    return lc_documents


def init_all_metrics() -> list[MetricWithLLM]:
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        harmfulness,
    ]

    ragas_llm = llm_factory()
    embed_model = OpenAIEmbeddings()

    # NOTE: error from RAGAS, need this fix
    for m in metrics:
        m.__setattr__("llm", ragas_llm)
        m.__setattr__("embeddings", embed_model)

    return metrics


# def compute_embedding(documents: list[Document]) -> DocumentStore:
#     embeddings = OpenAIEmbeddings()
#     docstore = DocumentStore()
#     for doc in documents:
#         doc.embedding = embeddings.embed_query(doc.page_content)
#         docstore.add_document(doc)
#     return docstore


def generate_synthetic_test_question(
    lc_documents: list[Document],
    distributions: dict = {simple: 0.5, multi_context: 0.4, reasoning: 0.1},
    test_size: int = 10,
) -> TestDataset:
    # generator with openai models
    generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
    critic_llm = ChatOpenAI(model="gpt-4")
    embeddings = OpenAIEmbeddings()

    generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)

    # NOTE: use generator.generate_with_llamaindex_docs if you use llama-index as document loader
    testset = generator.generate_with_langchain_docs(
        documents=lc_documents, test_size=test_size, distributions=distributions
    )
    return testset


def testset_to_csv(
    testset: TestDataset,
    filepath: str = "./testset.csv",
) -> None:
    df = testset.to_pandas()
    df.to_csv(filepath, index=False)
    return


if __name__ == "__main__":
    lc_documents = download_semantic_scholar_document()
    testset = generate_synthetic_test_question(
        lc_documents,
        test_size=20,
        distributions={simple: 0.4, multi_context: 0.4, reasoning: 0.2},
    )
    testset_to_csv(testset, filepath="./llm_topic_testset.csv")
