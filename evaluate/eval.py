import dotenv

dotenv.load_dotenv()

import os
import argparse
import pandas as pd
from ragas import evaluate
from ragas.evaluation import Result
from ragas.testset.generator import TestsetGenerator
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.core import BaseQueryEngine
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
from langchain.document_loaders.base import Document as LCDocument
from llama_index.schema import Document as LlamaIndexDocument
from ragas.metrics.critique import harmfulness
from llama_index.query_engine import MultiStepQueryEngine, SubQuestionQueryEngine
from llama_index.indices.query.query_transform.base import StepDecomposeQueryTransform
from llama_index.llms.openai import OpenAI
from llama_index.llm_predictor import LLMPredictor

from datasets import Dataset


def download_semantic_scholar_document(
    query: str = "Large Language Models", top_k: int = 10
) -> tuple[list[LCDocument], list[LlamaIndexDocument]]:
    from llama_index import download_loader

    SemanticScholarReader = download_loader("SemanticScholarReader")
    loader = SemanticScholarReader()
    query_space = query
    llamaindex_documents = loader.load_data(
        query=query_space, full_text=True, limit=top_k
    )
    lc_documents = [doc.to_langchain_format() for doc in llamaindex_documents]
    return lc_documents, llamaindex_documents


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


def index_builder(llamaindex_documents: LlamaIndexDocument) -> VectorStoreIndex:
    vector_index = VectorStoreIndex.from_documents(
        llamaindex_documents,
        service_context=ServiceContext.from_defaults(chunk_size=512),
    )
    return vector_index


def simple_query_engine_builder(vector_index: VectorStoreIndex, top_k: int = 5) -> any:
    query_engine = vector_index.as_query_engine(top_k=top_k)
    return query_engine


def multi_step_query_engine_builder(
    vector_index: VectorStoreIndex,
    index_summary: str = "Used to answer questions about the Large Language Models (LLM)",
    num_steps: int = 3,
    top_k: int = 5,
    model_name: str = "gpt-3.5-turbo",  # NOTE: or GPT4
    llm_temp: float = 0.2,
):
    llm = OpenAI(temperature=llm_temp, model=model_name)
    step_decompose_transform = StepDecomposeQueryTransform(
        verbose=True,
        llm_predictor=LLMPredictor(llm=llm),
    )
    query_engine = vector_index.as_query_engine(top_k=top_k)
    query_engine = MultiStepQueryEngine(
        query_engine=query_engine,
        query_transform=step_decompose_transform,
        index_summary=index_summary,
        num_steps=num_steps,
    )
    return query_engine


def load_testset_from_csv(filepath: str) -> TestDataset:
    df = pd.read_csv(filepath)
    _test_dataset = Dataset.from_pandas(df)
    testset = TestDataset(test_data=_test_dataset)
    return testset


def load_answer_from_csv(filepath: str) -> Dataset:
    df = pd.read_csv(filepath)
    dataset = Dataset.from_pandas(df)
    return dataset


def generate_answer(
    query_engine: BaseQueryEngine,
    testset: TestDataset,
    output_filepath: str = "./answer.csv",
) -> Dataset:
    contexts = []
    answers = []
    ground_truths = []
    questions = []
    for record in testset._to_records():
        question = record["question"]
        gt = record["ground_truth"]
        questions.append(question)
        ground_truths.append(gt)

        response = query_engine.query(question)
        contexts.append([x.node.get_content() for x in response.source_nodes])
        answers.append(str(response))

    # NOTE: schema from RAGAS
    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )
    ds.to_csv(output_filepath, index=False)
    return ds


def rag_evaluate(
    answer_dataset: Dataset,
    output_filepath: str = "./evaluate_result.csv",
) -> Result:
    metrics = init_all_metrics()
    result = evaluate(answer_dataset, metrics)
    df = result.to_pandas()
    df.to_csv(output_filepath, index=False)
    print(result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument(
        "--testset_csv",
        type=str,
        default="./llm_topic_testset.csv",
        help="Path to the testset CSV file",
    )
    parser.add_argument(
        "--output_filepath",
        type=str,
        default="./run_llm_nstep1/answer.csv",
        help="Path to the output CSV file",
    )
    parser.add_argument(
        "--answer_csv",
        type=str,
        default="./run_llm_nstep1/answer.csv",
        help="Path to the answer CSV file",
    )
    parser.add_argument(
        "--eval_output_filepath",
        type=str,
        default="./run_llm_nstep1/eval_result.csv",
        help="Path to the evaluation result CSV file",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=3,
        help="Number of steps for the multi-step query engine",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top K results for the multi-step query engine",
    )
    return parser.parse_args()


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def main(args):
    root_folder = os.path.dirname(args.output_filepath)
    create_folder(root_folder)

    testset = load_testset_from_csv(args.testset_csv)

    _, llamaindex_documents = download_semantic_scholar_document()
    vector_index = index_builder(llamaindex_documents)
    query_engine = multi_step_query_engine_builder(
        vector_index=vector_index, num_steps=args.num_steps, top_k=args.top_k
    )

    answers = generate_answer(
        query_engine, testset, output_filepath=args.output_filepath
    )

    answer_dataset = load_answer_from_csv(args.answer_csv)

    result = rag_evaluate(
        answer_dataset=answers, output_filepath=args.eval_output_filepath
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)


# if __name__ == "__main__":
# NOTE: load testset
# testset = load_testset_from_csv("./testset.csv")

# _, llamaindex_documents = download_semantic_scholar_document()
# vector_index = index_builder(llamaindex_documents)
# query_engine = simple_query_engine_builder(
#     vector_index=vector_index,
# )

# answers = generate_answer(
#     query_engine,
#     testset,
#     output_filepath="./run1/answer_simple_query_engine.csv",
# )
# # NOTE: load answer
# answer_dataset = load_answer_from_csv("./run1/answer_simple_query_engine.csv")
# # NOTE: evaluate
# result = rag_evaluate(
#     answer_dataset=answers,
#     output_filepath="./run1/evaluate_result_simple_query_engine.csv",
# )

# # NOTE: load testset
# testset = load_testset_from_csv("./llm_topic_testset.csv")

# _, llamaindex_documents = download_semantic_scholar_document()
# vector_index = index_builder(llamaindex_documents)
# query_engine = simple_query_engine_builder(
#     vector_index=vector_index,
# )

# answers = generate_answer(
#     query_engine,
#     testset,
#     output_filepath="./run_llm_nstep1/answer.csv",
# )
# # NOTE: load answer
# answer_dataset = load_answer_from_csv("./run_llm_nstep1/answer.csv")
# # NOTE: evaluate
# result = rag_evaluate(
#     answer_dataset=answers,
#     output_filepath="./run_llm_nstep1/eval_result.csv",
# )
