import os
import logging
from langsmith import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ls_client = Client()

DATASET_NAME = "NCO_Career_Advisor_Benchmark"

# Define benchmark evaluation cases
BENCHMARK_CASES = [
    {
        "inputs": {"query": "what jobs are there in space that have fixing things?"},
        "outputs": {
            "reference": "Aerospace mechanics, avionics technicians, and aircraft instrument fitters repair and maintain flight systems."
        }
    },
    {
        "inputs": {"query": "I want to work with computers and build software"},
        "outputs": {
            "reference": "Software developers, systems analysts, and computer programmers design and build software application systems."
        }
    },
    {
        "inputs": {"query": "careers related to flying and controlling aircraft"},
        "outputs": {
            "reference": "Aircraft pilots, flight engineers, air traffic controllers, and flight navigators."
        }
    }
]

def upload_dataset():
    # Check if dataset already exists
    if ls_client.has_dataset(dataset_name=DATASET_NAME):
        logger.info(f"Dataset '{DATASET_NAME}' already exists. Updating/Skipping creation.")
        dataset = ls_client.read_dataset(dataset_name=DATASET_NAME)
    else:
        logger.info(f"Creating new LangSmith dataset: '{DATASET_NAME}'")
        dataset = ls_client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Benchmark test cases for NCO Career Advisor RAG agent regression testing."
        )

    # Upload test cases
    for case in BENCHMARK_CASES:
        # Check if example already exists in the dataset to avoid duplicates
        existing_examples = ls_client.list_examples(dataset_id=dataset.id)
        existing_queries = [e.inputs.get("query") for e in existing_examples]
        
        if case["inputs"]["query"] not in existing_queries:
            ls_client.create_example(
                inputs=case["inputs"],
                outputs=case["outputs"],
                dataset_id=dataset.id
            )
            logger.info(f"Added test case: '{case['inputs']['query']}'")

    logger.info("Dataset setup complete.")

if __name__ == "__main__":
    upload_dataset()