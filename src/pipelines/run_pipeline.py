import os
from src.pipelines.zenml_pipeline import llm_support_pipeline

def main():
    # Respect any shell-exported value; otherwise default to 5001
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")

    run = llm_support_pipeline()  # This triggers execution in your ZenML version
    print(f"Pipeline finished. Run name={run.name} status={run.status}")

if __name__ == "__main__":
    main()
