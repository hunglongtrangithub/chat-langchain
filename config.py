import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr
from dotenv import load_dotenv

# NOTE: The environment variable from .env will be overridden by the one in the system environment
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8",
    )
    # Declared environment variables
    openai_api_key: SecretStr
    replicate_api_token: SecretStr
    groq_api_key: SecretStr
    hf_token: SecretStr
    port: int = 8080


settings = Settings()  # type: ignore
print(settings.model_dump_json(indent=2))

if __name__ == "__main__":
    print(settings.model_dump())
    print(settings.openai_api_key.get_secret_value() == os.getenv("OPENAI_API_KEY"))
