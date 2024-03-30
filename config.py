from pydantic import BaseSettings


class Settings(BaseSettings):  # type: ignore

    # Declared environment variables
    openai_api_key: str
    langchain_tracing_v2: bool
    langchain_endpoint: str
    langchain_api_key: str
    langchain_project: str
    replicate_api_token: str
    hf_token: str
    port: int

    class Config:  # type: ignore
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()  # type: ignore
if __name__ == "__main__":
    print(settings.port)
