import os
import json
import logging
from langsmith import Client
from langsmith.evaluation import evaluate, EvaluationResult
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from retriever import NCORetriever
from rag import NCORAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ls_client = Client()
DATASET_NAME = "NCO_Career_Advisor_Benchmark"

# Initialize Evaluator LLM (JSON mode enabled for simple outputs)
ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
evaluator_llm = ChatOllama(
    model="gemma:2b", 
    base_url=ollama_url, 
    temperature=0.0,
    format="json"
)

# Initialize Agent
retriever = NCORetriever()
agent = NCORAG(retriever)

def predict_target(inputs: dict) -> dict:
    query = inputs["query"]
    result = agent.generate_answer(query)
    
    contexts = [d.get("description", "") for d in result.get("sources", []) if d.get("description")]
    answer = result.get("answer", "")
    
    if not answer.strip():
        answer = "No response generated."
    if not contexts:
        contexts = ["No contexts retrieved."]

    return {
        "output": answer,
        "contexts": contexts
    }

# --- CUSTOM LIGHTWEIGHT EVALUATORS FOR SMALL LLMS ---

def evaluate_faithfulness(answer: str, contexts: list) -> float:
    """Checks if the answer is grounded in the retrieved context."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict judge. Check if the Answer is based ONLY on the Context. 
Output ONLY a JSON object with a single key 'score' and a float value 1.0 (if yes) or 0.0 (if no/hallucinated)."""),
        ("user", "Context: {context}\n\nAnswer: {answer}")
    ])
    try:
        context_str = "\n".join(contexts)
        res = (prompt | evaluator_llm).invoke({"context": context_str, "answer": answer})
        data = json.loads(res.content)
        return float(data.get("score", 0.0))
    except Exception as e:
        logger.warning(f"Faithfulness parsing failed: {e}")
        return 0.0

def evaluate_relevance(query: str, answer: str) -> float:
    """Checks if the answer directly addresses the user's query."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict judge. Check if the Answer directly addresses the User Query.
Output ONLY a JSON object with a single key 'score' and a float value 1.0 (if highly relevant) or 0.0 (if not relevant)."""),
        ("user", "Query: {query}\n\nAnswer: {answer}")
    ])
    try:
        res = (prompt | evaluator_llm).invoke({"query": query, "answer": answer})
        data = json.loads(res.content)
        return float(data.get("score", 0.0))
    except Exception as e:
        logger.warning(f"Relevance parsing failed: {e}")
        return 0.0

# --- MAIN EVALUATOR ADAPTER ---

def custom_evaluator(run, example) -> list[dict]:
    query = example.inputs.get("query", "")
    prediction = run.outputs.get("output", "No response generated.")
    contexts = run.outputs.get("contexts", ["No contexts retrieved."])

    # Run the lightweight checks
    faithfulness_score = evaluate_faithfulness(prediction, contexts)
    relevance_score = evaluate_relevance(query, prediction)

    return [
        {"key": "custom_faithfulness", "score": faithfulness_score},
        {"key": "custom_answer_relevancy", "score": relevance_score}
    ]

def execute_regression():
    logger.info(f"--- STARTING REGRESSION TEST ON DATASET: {DATASET_NAME} ---")
    
    evaluate(
        predict_target,
        data=DATASET_NAME,
        evaluators=[custom_evaluator],
        experiment_prefix="gemma2b-custom-eval",
        metadata={"model": "gemma:2b", "pipeline": "CRAG Custom Eval"}
    )
    
    logger.info("--- REGRESSION TEST COMPLETE ---")

if __name__ == "__main__":
    execute_regression()