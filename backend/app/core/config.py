from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    vector_store_path: str = "data/vector_store"
    collection_name: str = "appliance_manuals"
    embedding_model: str = "all-MiniLM-L6-v2"
    ollama_model: str = "llama3.2"
    ollama_host: str = "http://localhost:11434"
    cors_origins: list[str] = ["http://localhost:8501"]
    retrieval_k: int = 3

    class Config:
        env_file = ".env"


settings = Settings()
