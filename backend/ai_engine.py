
import logging
import os
from typing import Any
from urllib.parse import quote

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("chatbot_ai.engine")

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
).strip()

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
)

MAX_MESSAGE_LENGTH = 12000
MAX_HISTORY_MESSAGES = 12
MAX_HISTORY_ITEM_LENGTH = 6000
MAX_OUTPUT_TOKENS = 2048


# --------------------------------------------------
# CUSTOM ERRORS
# --------------------------------------------------

class AIEngineError(Exception):
    """Base error for the AI engine."""


class AIConfigurationError(AIEngineError):
    """Raised when AI configuration is missing or invalid."""


class AIProviderError(AIEngineError):
    """Raised when the AI provider cannot return a response."""


# --------------------------------------------------
# CHATBOT AI PERSONALITY AND INSTRUCTIONS
# --------------------------------------------------

SYSTEM_INSTRUCTION = """
You are ChatBot AI, created by Neliseka Mdunge.

IDENTITY:
- If asked who created ChatBot AI, answer:
  "ChatBot AI was created by Neliseka Mdunge."
- You are an artificial intelligence assistant.
- Never pretend to be human or claim real-life experiences.
- You are a general-purpose assistant, not limited to a single subject.

PERSONALITY:
- Always be friendly, patient, respectful and welcoming.
- Never insult, bully, mock, threaten or humiliate users.
- Never swear at users or use offensive language against them.
- Treat people respectfully regardless of their knowledge or background.
- Listen carefully when users discuss everyday personal experiences.
- Respond with empathy and practical support.
- Never shame users for asking basic or repeated questions.

ACCURACY AND HONESTY:
- Never deliberately invent facts, sources, links, dates, scores,
  statistics, software versions or technical instructions.
- Never claim that an answer has been independently verified unless
  a real verification source or tool was used.
- Clearly state when information is uncertain or incomplete.
- Ask for clarification when important details are missing.
- Separate confirmed facts from estimates, predictions and opinions.
- Never fabricate citations or pretend to have browsed the internet.
- Do not claim that code was tested, deployed or executed unless
  an actual tool performed that action.
- For current information, explain that a live search integration
  is needed when one has not been provided.
- For medical, legal, financial or other high-impact topics,
  be especially careful, explain limitations and encourage
  appropriate qualified help.

GENERAL CAPABILITIES:
Help users learn, understand and solve problems involving:
- General knowledge and everyday questions.
- Python and other programming languages.
- Coding lessons, debugging and project development.
- School subjects, homework and study preparation.
- Website and application development.
- GitHub, Git and software deployment.
- Termux, Linux commands and Ubuntu.
- Authorized VPS administration and server maintenance.
- Legitimate VPN protocols and configuration files.
- Telegram bots and authorized WhatsApp bot development.
- Authorized 3X-UI server administration.
- Personal organisation, planning and writing.
- Sports information when reliable current data is available.
- Image and video creation planning and editing assistance.

TECHNICAL SAFETY:
- Explain technical instructions step by step.
- Warn users about commands that can delete data or damage systems.
- Never request passwords, API keys or private credentials in chat.
- Do not expose secrets in generated code.
- Only help with systems and servers the user is authorized to manage.
- Do not help bypass paid access, network restrictions or security
  controls.
- Use official documentation where available.
- Do not claim that every application supports every configuration.

SPORTS:
- Never invent fixtures, scores, team news or player lineups.
- Distinguish confirmed lineups from predicted lineups.
- Explain when live sports data is unavailable.

DOWNLOAD LINKS:
- Never invent a download URL.
- Prefer official websites and trusted app stores.
- Clearly state when a link has not been verified.

IMAGE AND VIDEO:
- Do not claim to have generated or edited media unless the relevant
  image or video tool has actually completed that operation.
- You may help users plan edits, write prompts and create instructions.

CONVERSATION:
- Understand casual messages and natural questions.
- Answer the user's actual request instead of forcing every message
  into a technical format.
- Use clear language and explain difficult concepts simply.
- Adapt the detail level to the user's needs.
- Be helpful without pretending to know everything.

IMPORTANT:
Being confident does not make an answer correct.
When you do not know something, say so honestly.
"""


# --------------------------------------------------
# CONVERSATION HISTORY
# --------------------------------------------------

def prepare_history(
    history: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """
    Convert application conversation history into Gemini format.

    Accepted application roles:
      user
      assistant
    """

    if not history:
        return []

    prepared = []

    for item in history[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if not isinstance(content, str):
            continue

        content = content.strip()

        if not content:
            continue

        content = content[:MAX_HISTORY_ITEM_LENGTH]

        if role == "user":
            gemini_role = "user"
        elif role == "assistant":
            gemini_role = "model"
        else:
            continue

        prepared.append({
            "role": gemini_role,
            "parts": [
                {"text": content}
            ]
        })

    return prepared


# --------------------------------------------------
# AI STATUS
# --------------------------------------------------

def get_ai_status() -> dict[str, Any]:
    """Return safe AI configuration information."""

    configured = bool(GEMINI_API_KEY)

    return {
        "provider": "Google Gemini",
        "model": GEMINI_MODEL,
        "configured": configured,
        "status": (
            "ready"
            if configured
            else "waiting_for_api_key"
        )
    }


# --------------------------------------------------
# GEMINI RESPONSE EXTRACTION
# --------------------------------------------------

def extract_response(data: dict[str, Any]) -> str:
    """Safely extract generated text from the API response."""

    prompt_feedback = data.get("promptFeedback", {})

    if prompt_feedback.get("blockReason"):
        raise AIProviderError(
            "The request was blocked by the provider's safety filters."
        )

    candidates = data.get("candidates", [])

    if not candidates:
        raise AIProviderError(
            "The AI provider did not return a usable answer."
        )

    candidate = candidates[0]

    finish_reason = candidate.get("finishReason", "")

    if finish_reason == "SAFETY":
        raise AIProviderError(
            "The response was blocked by the provider's safety filters."
        )

    content = candidate.get("content", {})
    parts = content.get("parts", [])

    response_text = "\n".join(
        part["text"]
        for part in parts
        if isinstance(part, dict)
        and isinstance(part.get("text"), str)
        and part.get("text").strip()
        and not part.get("thought", False)
    ).strip()

    if not response_text:
        raise AIProviderError(
            "The AI provider returned an empty response."
        )

    return response_text


# --------------------------------------------------
# MAIN AI FUNCTION
# --------------------------------------------------

async def generate_reply(
    message: str,
    history: list[dict[str, Any]] | None = None
) -> str:
    """
    Generate a ChatBot AI response.

    Args:
        message:
            The latest user message.

        history:
            Optional previous messages:
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello!"},
            ]

    Returns:
        Generated assistant response as text.

    Raises:
        AIConfigurationError
        AIProviderError
        ValueError
    """

    if not isinstance(message, str):
        raise ValueError("Message must be text.")

    message = message.strip()

    if not message:
        raise ValueError("Message cannot be empty.")

    if len(message) > MAX_MESSAGE_LENGTH:
        raise ValueError(
            f"Message is too long. Maximum is "
            f"{MAX_MESSAGE_LENGTH} characters."
        )

    if not GEMINI_API_KEY:
        raise AIConfigurationError(
            "The Gemini API key has not been configured."
        )

    if not GEMINI_MODEL:
        raise AIConfigurationError(
            "The Gemini model name is missing."
        )

    model_name = quote(GEMINI_MODEL, safe="-._")

    endpoint = (
        f"{GEMINI_API_URL}{model_name}:generateContent"
    )

    previous_messages = prepare_history(history)

    conversation = previous_messages + [
        {
            "role": "user",
            "parts": [
                {"text": message}
            ]
        }
    ]

    payload = {
        "systemInstruction": {
            "parts": [
                {"text": SYSTEM_INSTRUCTION}
            ]
        },
        "contents": conversation,
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_LOW_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_LOW_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_LOW_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_LOW_AND_ABOVE"
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.9,
            "maxOutputTokens": MAX_OUTPUT_TOKENS
        }
    }

    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json"
    }

    timeout = httpx.Timeout(
        connect=10.0,
        read=60.0,
        write=30.0,
        pool=10.0
    )

    try:
        async with httpx.AsyncClient(
            timeout=timeout
        ) as client:

            response = await client.post(
                endpoint,
                headers=headers,
                json=payload
            )

        if response.status_code == 401:
            raise AIProviderError(
                "The provider rejected the API key. "
                "Check your server configuration."
            )

        if response.status_code == 403:
            raise AIProviderError(
                "The provider denied access. "
                "Check API permissions and account access."
            )

        if response.status_code == 404:
            raise AIProviderError(
                "The selected model was not found or is unavailable "
                "for this API key. Check GEMINI_MODEL."
            )

        if response.status_code == 429:
            raise AIProviderError(
                "The AI service rate limit or quota was reached. "
                "Please try again later."
            )

        if response.status_code >= 500:
            raise AIProviderError(
                "The AI provider is temporarily unavailable."
            )

        if response.status_code >= 400:
            raise AIProviderError(
                f"The AI provider returned HTTP "
                f"{response.status_code}."
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise AIProviderError(
                "The AI provider returned an invalid response."
            ) from exc

        answer = extract_response(data)

        logger.info("AI response generated successfully.")

        return answer

    except AIProviderError:
        raise

    except httpx.TimeoutException as exc:
        logger.warning("AI provider request timed out.")

        raise AIProviderError(
            "The AI service took too long to respond. "
            "Please try again."
        ) from exc

    except httpx.HTTPError as exc:
        logger.error(
            "AI provider network request failed: %s",
            type(exc).__name__
        )

        raise AIProviderError(
            "Unable to connect to the AI service."
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected AI engine error.")

        raise AIProviderError(
            "An unexpected error occurred while generating a response."
        ) from exc
