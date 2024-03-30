import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


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
if __name__ == "__main__":
    print(settings.model_dump())
    print(settings.groq_api_key.get_secret_value() == os.getenv("GROQ_API_KEY"))
