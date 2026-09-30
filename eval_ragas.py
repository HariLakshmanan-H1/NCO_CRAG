import os
import logging
from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    LLMContextPrecisionWithReference
)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langsmith import Client, traceable
from retriever import NCORetriever
from rag import NCORAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LangSmith Client
ls_client = Client()

# 1. Initialize Local LLM & Embeddings with Ragas Wrappers and JSON mode
ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
raw_llm = ChatOllama(
    model="gemma:2b", 
    base_url=ollama_url, 
    temperature=0.0,
    format="json"  # <-- Added to force strict JSON output for Ragas
)
raw_embeddings = OllamaEmbeddings(model="gemma:2b", base_url=ollama_url)

evaluator_llm = LangchainLLMWrapper(raw_llm)
evaluator_embeddings = LangchainEmbeddingsWrapper(raw_embeddings)

# Instantiate Ragas Metric Classes using wrapped instances
faithfulness_metric = Faithfulness(llm=evaluator_llm)
answer_relevancy_metric = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
context_precision_metric = LLMContextPrecisionWithReference(llm=evaluator_llm)

test_cases = [
    {
        "user_input": "what jobs are there in space that have fixing things?",
        "reference": "Aerospace mechanics, avionics technicians, and aircraft instrument fitters repair and maintain flight systems."
    },
    {
        "user_input": "I want to work with computers and build software",
        "reference": "Software developers, systems analysts, and computer programmers design and build software application systems."
    }
]

# Wrap execution with @traceable so the evaluation session is tracked in LangSmith
@traceable(name="Ragas_Evaluation_Batch", project_name="NCO_Career_Advisor")
def run_evaluation_and_log():
    retriever = NCORetriever()
    agent = NCORAG(retriever)

    user_inputs = []
    responses = []
    retrieved_contexts = []
    references = []

    logger.info("--- EXECUTING AGENT TRACES ---")
    for case in test_cases:
        query = case["user_input"]
        result = agent.generate_answer(query)

        docs = [d.get("description", "") for d in result.get("sources", []) if d.get("description")]
        answer = result.get("answer", "")

        # Fallback values to prevent empty sequence issues
        if not answer.strip():
            answer = "No response generated."
        if not docs:
            docs = ["No contexts retrieved."]

        user_inputs.append(query)
        responses.append(answer)
        retrieved_contexts.append(docs)
        references.append(case["reference"])

    # Build Dataset for Ragas
    data = {
        "user_input": user_inputs,
        "response": responses,
        "retrieved_contexts": retrieved_contexts,
        "reference": references
    }
    dataset = Dataset.from_dict(data)

    logger.info("--- COMPUTING RAGAS METRICS ---")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness_metric, answer_relevancy_metric, context_precision_metric],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    print("\n================ RAGAS RESULTS ================")
    print(results)
    print("===============================================")

    # Push Ragas scores to LangSmith as Feedback Metrics
    for metric_name, score in results.items():
        if isinstance(score, (int, float)):
            ls_client.create_feedback(
                run_id=None,  # Logs as a dataset/project-level score trace
                key=f"ragas_{metric_name}",
                score=float(score),
                comment=f"Automated Ragas evaluation score for {metric_name}"
            )
            logger.info(f"Logged {metric_name}: {score} to LangSmith")

if __name__ == "__main__":
    run_evaluation_and_log()