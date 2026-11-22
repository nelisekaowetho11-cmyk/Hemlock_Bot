
import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from the project .env file.
load_dotenv()


def read_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable safely."""
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1", "true", "yes", "on"
    }


def read_int(
    name: str,
    default: int,
    minimum: int,
    maximum: int
) -> int:
    """Read and validate an integer environment variable."""
    value = os.getenv(name)

    if value is None:
        return default

    try:
        number = int(value.strip())
    except ValueError:
        return default

    if not minimum <= number <= maximum:
        return default

    return number


def read_list(name: str) -> list[str]:
    """Read a comma-separated environment variable."""
    value = os.getenv(name, "")

    return [
        item.strip().rstrip("/")
        for item in value.split(",")
        if item.strip()
    ]


@dataclass(frozen=True)
class Settings:
    # Application information
    app_name: str
    app_creator: str
    app_version: str
    environment: str
    debug: bool
    log_level: str

    # AI provider
    gemini_api_key: str
    gemini_model: str
    gemini_api_base_url: str
    gemini_timeout_seconds: int
    gemini_max_output_tokens: int

    # Conversation limits
    max_message_length: int
    max_history_messages: int
    max_history_item_length: int

    # Frontend security
    allowed_origins: list[str]

    @classmethod
    def from_environment(cls):
        environment = os.getenv(
            "CHATBOT_ENV",
            "development"
        ).strip().lower()

        log_level = os.getenv(
            "LOG_LEVEL",
            "INFO"
        ).strip().upper()

        valid_log_levels = {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL"
        }

        if log_level not in valid_log_levels:
            log_level = "INFO"

        origins = read_list("CHATBOT_ALLOWED_ORIGINS")

        if not origins and environment in {
            "development", "dev", "local"
        }:
            origins = [
                "http://localhost:3000",
                "http://localhost:5173",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5173"
            ]

        return cls(
            # Application information
            app_name="ChatBot AI",
            app_creator="Neliseka Mdunge",
            app_version="0.1.0",
            environment=environment,
            debug=read_bool(
                "CHATBOT_DEBUG",
                environment != "production"
            ),
            log_level=log_level,

            # AI provider
            gemini_api_key=os.getenv(
                "GEMINI_API_KEY", ""
            ).strip(),

            gemini_model=os.getenv(
                "GEMINI_MODEL",
                "gemini-3.8-flash"
            ).strip(),

            gemini_api_base_url=os.getenv(
                "GEMINI_API_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/models"
            ).strip().rstrip("/"),

            gemini_timeout_seconds=read_int(
                "GEMINI_TIMEOUT_SECONDS",
                60,
                5,
                180
            ),

            gemini_max_output_tokens=read_int(
                "GEMINI_MAX_OUTPUT_TOKENS",
                2048,
                128,
                8192
            ),

            # Conversation limits
            max_message_length=read_int(
                "CHATBOT_MAX_MESSAGE_LENGTH",
                12000,
                100,
                100000
            ),

            max_history_messages=read_int(
                "CHATBOT_MAX_HISTORY_MESSAGES",
                12,
                0,
                100
            ),

            max_history_item_length=read_int(
                "CHATBOT_MAX_HISTORY_ITEM_LENGTH",
                6000,
                100,
                20000
            ),

            # Frontend security
            allowed_origins=origins
        )


# One shared configuration instance.
settings = Settings.from_environment()


def get_public_settings() -> dict:
    """
    Return non-sensitive settings.

    Never return API keys or other secrets.
    """
    return {
        "app_name": settings.app_name,
        "creator": settings.app_creator,
        "version": settings.app_version,
        "environment": settings.environment,
        "ai_provider": "Google Gemini",
        "ai_model": settings.gemini_model,
        "ai_configured": bool(
            settings.gemini_api_key
        )
    }
