from main import run_pipeline
from raindrop_engine import RaindropEngine

if __name__ == "__main__":
    run_pipeline(mode="train", engine_class=RaindropEngine)
