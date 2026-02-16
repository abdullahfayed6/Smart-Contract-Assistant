from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_chat_model: str = Field(
        default="gpt-4o-mini",
        alias="OPENAI_CHAT_MODEL",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-large",
        alias="OPENAI_EMBEDDING_MODEL",
    )
    openai_rerank_model: str = Field(
        default="gpt-4o-mini",
        alias="OPENAI_RERANK_MODEL",
    )

    chroma_dir: str = Field(default="data/chroma", alias="CHROMA_DIR")
    upload_dir: str = Field(default="data/uploads", alias="UPLOAD_DIR")
    processed_dir: str = Field(default="data/processed", alias="PROCESSED_DIR")

    default_top_k: int = Field(default=20, alias="DEFAULT_TOP_K")
    rerank_top_k: int = Field(default=8, alias="RERANK_TOP_K")
    retrieval_min_score: float = Field(default=0.3, alias="RETRIEVAL_MIN_SCORE")
    list_intent_top_k: int = Field(default=50, alias="LIST_INTENT_TOP_K")
    list_intent_rerank_top_k: int = Field(default=25, alias="LIST_INTENT_RERANK_TOP_K")
    list_intent_min_score: float = Field(default=0.2, alias="LIST_INTENT_MIN_SCORE")

    max_chunk_chars: int = Field(default=900, alias="MAX_CHUNK_CHARS")
    chunk_overlap_chars: int = Field(default=180, alias="CHUNK_OVERLAP_CHARS")

    @property
    def resolved_api_key(self) -> str:
        if self.openai_api_key:
            return self.openai_api_key

        key_file = Path("openai_api.txt")
        if key_file.exists():
            value = key_file.read_text(encoding="utf-8").strip()
            if value:
                return value

        raise ValueError(
            "OpenAI API key is missing. Set OPENAI_API_KEY in .env or place key in openai_api.txt."
        )

    def ensure_dirs(self) -> None:
        Path(self.chroma_dir).mkdir(parents=True, exist_ok=True)
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
