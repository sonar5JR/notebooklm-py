"""NotebookLM MCP Server - Exposes NotebookLM API as MCP tools.

This MCP server wraps the notebooklm-py library to provide full
programmatic access to Google NotebookLM via the Model Context Protocol.
"""

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# Add src to path so we can import notebooklm
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent / "src"))

from notebooklm import NotebookLMClient

logger = logging.getLogger(__name__)

# Global client reference
_client: NotebookLMClient | None = None

# ─── Safe download directory ─────────────────────────────────────────────────
# All artifact downloads are confined to this directory to prevent
# path-traversal attacks via prompt injection. Override with the
# NOTEBOOKLM_DOWNLOAD_DIR environment variable.
ALLOWED_DOWNLOAD_DIR = Path(
    os.environ.get("NOTEBOOKLM_DOWNLOAD_DIR", "./downloads")
).resolve()

# Sensitive directory patterns that must NEVER be written to
_SENSITIVE_DIRS = {
    ".ssh", ".gnupg", ".config", ".aws", ".azure", ".kube",
    "AppData", "System32", "SysWOW64", "Program Files",
    "Program Files (x86)", "Windows",
}


def _sanitize_output_path(output_path: str) -> Path:
    """Sanitize an output_path to prevent path-traversal attacks.

    Security guarantees:
    1. Rejects paths containing '..' traversal sequences
    2. Blocks writes to sensitive system directories
    3. Confines all writes to ALLOWED_DOWNLOAD_DIR
    4. Auto-creates the target directory if needed

    Raises:
        ValueError: If the path is unsafe or escapes the allowed directory.
    """
    # Step 1: Reject any raw traversal sequences before resolution
    if ".." in output_path:
        raise ValueError(
            f"Path traversal detected: output_path must not contain '..'. "
            f"Got: {output_path!r}"
        )

    # Step 2: Strip leading slashes / drive letters to force relative interpretation
    #         e.g. "C:\\Windows\\evil.exe" -> "Windows\\evil.exe"
    #         e.g. "/etc/passwd" -> "etc/passwd"
    cleaned = output_path
    # Remove Windows drive prefix like C:\ or D:\
    if len(cleaned) >= 2 and cleaned[1] == ":":
        cleaned = cleaned[2:]
    # Strip leading slashes/backslashes
    cleaned = cleaned.lstrip("/").lstrip("\\")

    if not cleaned:
        raise ValueError("output_path must not be empty after sanitization.")

    # Step 3: Resolve within the allowed directory
    safe_path = (ALLOWED_DOWNLOAD_DIR / cleaned).resolve()

    # Step 4: Verify the resolved path is still within ALLOWED_DOWNLOAD_DIR
    try:
        safe_path.relative_to(ALLOWED_DOWNLOAD_DIR)
    except ValueError:
        raise ValueError(
            f"Path escapes the allowed download directory. "
            f"Resolved: {safe_path}, Allowed: {ALLOWED_DOWNLOAD_DIR}"
        )

    # Step 5: Check against sensitive directory names
    for part in safe_path.parts:
        if part in _SENSITIVE_DIRS:
            raise ValueError(
                f"Refusing to write to sensitive directory: {part!r} "
                f"in path {safe_path}"
            )

    # Step 6: Ensure target directory exists
    safe_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Sanitized output path: %s -> %s", output_path, safe_path)
    return safe_path


async def get_client() -> NotebookLMClient:
    """Get or create the NotebookLM client singleton."""
    global _client
    if _client is None or not _client.is_connected:
        _client = await NotebookLMClient.from_storage()
        await _client.__aenter__()
    return _client


def _serialize(obj: Any) -> Any:
    """Serialize dataclass/object to JSON-safe dict."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialize(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if hasattr(obj, "__dict__"):
        return {k: _serialize(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    return str(obj)


# ─── Create MCP Server ───────────────────────────────────────────────────────

mcp = FastMCP(
    "NotebookLM",
    instructions="Full programmatic access to Google NotebookLM — notebooks, sources, chat, artifacts, and research.",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  NOTEBOOK TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def list_notebooks() -> str:
    """List all notebooks in your NotebookLM account.

    Returns a JSON array of notebook objects with id, title, source_count, and created/updated timestamps.
    """
    client = await get_client()
    notebooks = await client.notebooks.list()
    return json.dumps(_serialize(notebooks), indent=2)


@mcp.tool()
async def create_notebook(title: str) -> str:
    """Create a new notebook.

    Args:
        title: The title for the new notebook.

    Returns the created notebook object with its ID.
    """
    client = await get_client()
    nb = await client.notebooks.create(title)
    return json.dumps(_serialize(nb), indent=2)


@mcp.tool()
async def get_notebook(notebook_id: str) -> str:
    """Get details of a specific notebook.

    Args:
        notebook_id: The notebook ID.

    Returns the notebook object with full details.
    """
    client = await get_client()
    nb = await client.notebooks.get(notebook_id)
    return json.dumps(_serialize(nb), indent=2)


@mcp.tool()
async def rename_notebook(notebook_id: str, new_title: str) -> str:
    """Rename a notebook.

    Args:
        notebook_id: The notebook ID.
        new_title: The new title for the notebook.

    Returns the renamed notebook object.
    """
    client = await get_client()
    nb = await client.notebooks.rename(notebook_id, new_title)
    return json.dumps(_serialize(nb), indent=2)


@mcp.tool()
async def delete_notebook(notebook_id: str) -> str:
    """Delete a notebook.

    Args:
        notebook_id: The notebook ID to delete.

    Returns success status.
    """
    client = await get_client()
    result = await client.notebooks.delete(notebook_id)
    return json.dumps({"success": result})


@mcp.tool()
async def get_notebook_description(notebook_id: str) -> str:
    """Get AI-generated summary and suggested topics for a notebook.

    Args:
        notebook_id: The notebook ID.

    Returns the notebook description with summary and suggested topics/questions.
    """
    client = await get_client()
    desc = await client.notebooks.get_description(notebook_id)
    return json.dumps(_serialize(desc), indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  SOURCE TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def list_sources(notebook_id: str) -> str:
    """List all sources in a notebook.

    Args:
        notebook_id: The notebook ID.

    Returns a JSON array of source objects with id, title, type, and status.
    """
    client = await get_client()
    sources = await client.sources.list(notebook_id)
    return json.dumps(_serialize(sources), indent=2)


@mcp.tool()
async def add_url_source(notebook_id: str, url: str, wait: bool = True) -> str:
    """Add a URL source to a notebook. Also handles YouTube URLs automatically.

    Args:
        notebook_id: The notebook ID.
        url: The URL to add (webpage or YouTube link).
        wait: If True (default), wait for the source to be fully processed.

    Returns the created source object.
    """
    client = await get_client()
    source = await client.sources.add_url(notebook_id, url, wait=wait)
    return json.dumps(_serialize(source), indent=2)


@mcp.tool()
async def add_text_source(
    notebook_id: str, title: str, content: str, wait: bool = True
) -> str:
    """Add a text source (pasted/typed text) to a notebook.

    Args:
        notebook_id: The notebook ID.
        title: Title for the source.
        content: The text content to add.
        wait: If True (default), wait for the source to be fully processed.

    Returns the created source object.
    """
    client = await get_client()
    source = await client.sources.add_text(notebook_id, title, content, wait=wait)
    return json.dumps(_serialize(source), indent=2)


@mcp.tool()
async def delete_source(notebook_id: str, source_id: str) -> str:
    """Delete a source from a notebook.

    Args:
        notebook_id: The notebook ID.
        source_id: The source ID to delete.

    Returns success status.
    """
    client = await get_client()
    result = await client.sources.delete(notebook_id, source_id)
    return json.dumps({"success": result})


@mcp.tool()
async def rename_source(notebook_id: str, source_id: str, new_title: str) -> str:
    """Rename a source in a notebook.

    Args:
        notebook_id: The notebook ID.
        source_id: The source ID to rename.
        new_title: The new title for the source.

    Returns the updated source object.
    """
    client = await get_client()
    source = await client.sources.rename(notebook_id, source_id, new_title)
    return json.dumps(_serialize(source), indent=2)


@mcp.tool()
async def get_source_guide(notebook_id: str, source_id: str) -> str:
    """Get the source guide (AI-generated overview) for a specific source.

    Args:
        notebook_id: The notebook ID.
        source_id: The source ID.

    Returns the source guide text.
    """
    client = await get_client()
    source = await client.sources.get(notebook_id, source_id)
    return json.dumps(_serialize(source), indent=2)


@mcp.tool()
async def get_source_fulltext(notebook_id: str, source_id: str) -> str:
    """Get the full indexed text content of a source.

    Args:
        notebook_id: The notebook ID.
        source_id: The source ID.

    Returns the full text content that NotebookLM indexed from this source.
    """
    client = await get_client()
    fulltext = await client.sources.get_fulltext(notebook_id, source_id)
    return json.dumps(_serialize(fulltext), indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  CHAT TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def ask_notebook(
    notebook_id: str,
    question: str,
    source_ids: list[str] | None = None,
    conversation_id: str | None = None,
) -> str:
    """Ask a question to a notebook's AI assistant.

    The AI will answer based on the notebook's sources. You can optionally
    limit which sources to query and continue a previous conversation.

    Args:
        notebook_id: The notebook ID.
        question: The question to ask.
        source_ids: Optional list of specific source IDs to query. If None, uses all sources.
        conversation_id: Optional conversation ID for follow-up questions.

    Returns the AI answer, references, and conversation ID for follow-ups.
    """
    client = await get_client()
    result = await client.chat.ask(
        notebook_id,
        question,
        source_ids=source_ids,
        conversation_id=conversation_id,
    )
    return json.dumps(_serialize(result), indent=2)


@mcp.tool()
async def get_chat_history(
    notebook_id: str, limit: int = 20, conversation_id: str | None = None
) -> str:
    """Get the chat history for a notebook.

    Args:
        notebook_id: The notebook ID.
        limit: Maximum number of Q&A turns to retrieve (default: 20).
        conversation_id: Optional specific conversation ID. If None, uses most recent.

    Returns a list of (question, answer) pairs.
    """
    client = await get_client()
    history = await client.chat.get_history(
        notebook_id, limit=limit, conversation_id=conversation_id
    )
    return json.dumps(_serialize(history), indent=2)


@mcp.tool()
async def configure_chat(
    notebook_id: str,
    goal: str | None = None,
    response_length: str | None = None,
    custom_prompt: str | None = None,
) -> str:
    """Configure the chat persona and response settings for a notebook.

    Args:
        notebook_id: The notebook ID.
        goal: Chat persona — "default", "custom", or "learning_guide".
        response_length: Response length — "short", "default", or "long".
        custom_prompt: Custom system prompt (used when goal is "custom").

    Returns success status.
    """
    from notebooklm import ChatGoal, ChatResponseLength

    client = await get_client()

    goal_enum = None
    if goal:
        goal_map = {
            "default": ChatGoal.DEFAULT,
            "custom": ChatGoal.CUSTOM,
            "learning_guide": ChatGoal.LEARNING_GUIDE,
        }
        goal_enum = goal_map.get(goal.lower())

    length_enum = None
    if response_length:
        length_map = {
            "short": ChatResponseLength.SHORT,
            "default": ChatResponseLength.DEFAULT,
            "long": ChatResponseLength.LONG,
        }
        length_enum = length_map.get(response_length.lower())

    await client.chat.configure(
        notebook_id,
        goal=goal_enum,
        response_length=length_enum,
        custom_prompt=custom_prompt,
    )
    return json.dumps({"success": True})


# ═══════════════════════════════════════════════════════════════════════════════
#  ARTIFACT / CONTENT GENERATION TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def list_artifacts(notebook_id: str, artifact_type: str | None = None) -> str:
    """List all generated artifacts (content) in a notebook.

    Args:
        notebook_id: The notebook ID.
        artifact_type: Optional filter — "audio", "video", "report", "quiz",
            "flashcards", "infographic", "slide_deck", "data_table", "mind_map".

    Returns a JSON array of artifact objects with id, type, title, and status.
    """
    from notebooklm import ArtifactType

    client = await get_client()

    type_enum = None
    if artifact_type:
        type_map = {
            "audio": ArtifactType.AUDIO,
            "video": ArtifactType.VIDEO,
            "report": ArtifactType.REPORT,
            "quiz": ArtifactType.QUIZ,
            "flashcards": ArtifactType.FLASHCARDS,
            "infographic": ArtifactType.INFOGRAPHIC,
            "slide_deck": ArtifactType.SLIDE_DECK,
            "data_table": ArtifactType.DATA_TABLE,
            "mind_map": ArtifactType.MIND_MAP,
        }
        type_enum = type_map.get(artifact_type.lower())

    artifacts = await client.artifacts.list(notebook_id, artifact_type=type_enum)
    return json.dumps(_serialize(artifacts), indent=2)


@mcp.tool()
async def generate_audio(
    notebook_id: str,
    instructions: str | None = None,
    audio_format: str | None = None,
    audio_length: str | None = None,
    language: str = "en",
    wait: bool = True,
) -> str:
    """Generate an Audio Overview (podcast) from notebook sources.

    Args:
        notebook_id: The notebook ID.
        instructions: Custom instructions for the audio generation.
        audio_format: Format — "deep_dive", "brief", "critique", or "debate".
        audio_length: Length — "short", "medium", or "long".
        language: Language code (default: "en").
        wait: If True (default), wait for generation to complete.

    Returns the generation status with task_id.
    """
    from notebooklm import AudioFormat, AudioLength

    client = await get_client()

    fmt = None
    if audio_format:
        fmt_map = {
            "deep_dive": AudioFormat.DEEP_DIVE,
            "brief": AudioFormat.BRIEF,
            "critique": AudioFormat.CRITIQUE,
            "debate": AudioFormat.DEBATE,
        }
        fmt = fmt_map.get(audio_format.lower())

    length = None
    if audio_length:
        len_map = {
            "short": AudioLength.SHORT,
            "medium": AudioLength.MEDIUM,
            "long": AudioLength.LONG,
        }
        length = len_map.get(audio_length.lower())

    status = await client.artifacts.generate_audio(
        notebook_id,
        language=language,
        instructions=instructions,
        audio_format=fmt,
        audio_length=length,
    )

    if wait and status.task_id:
        await client.artifacts.wait_for_completion(notebook_id, status.task_id)

    return json.dumps(_serialize(status), indent=2)


@mcp.tool()
async def generate_video(
    notebook_id: str,
    instructions: str | None = None,
    video_style: str | None = None,
    language: str = "en",
    wait: bool = True,
) -> str:
    """Generate a Video Overview from notebook sources.

    Args:
        notebook_id: The notebook ID.
        instructions: Custom instructions for the video.
        video_style: Visual style — "classic", "whiteboard", "kawaii", "anime",
            "abstract", "retro", "nature", "scifi", "watercolor".
        language: Language code (default: "en").
        wait: If True (default), wait for generation to complete.

    Returns the generation status.
    """
    from notebooklm import VideoStyle

    client = await get_client()

    style = None
    if video_style:
        style_map = {
            "classic": VideoStyle.CLASSIC,
            "whiteboard": VideoStyle.WHITEBOARD,
            "kawaii": VideoStyle.KAWAII,
            "anime": VideoStyle.ANIME,
            "abstract": VideoStyle.ABSTRACT,
            "retro": VideoStyle.RETRO,
            "nature": VideoStyle.NATURE,
            "scifi": VideoStyle.SCIFI,
            "watercolor": VideoStyle.WATERCOLOR,
        }
        style = style_map.get(video_style.lower())

    status = await client.artifacts.generate_video(
        notebook_id,
        language=language,
        instructions=instructions,
        video_style=style,
    )

    if wait and status.task_id:
        await client.artifacts.wait_for_completion(notebook_id, status.task_id)

    return json.dumps(_serialize(status), indent=2)


@mcp.tool()
async def generate_report(
    notebook_id: str,
    report_format: str = "briefing_doc",
    custom_prompt: str | None = None,
    extra_instructions: str | None = None,
    language: str = "en",
) -> str:
    """Generate a report artifact from notebook sources.

    Args:
        notebook_id: The notebook ID.
        report_format: Format — "briefing_doc", "study_guide", "blog_post", or "custom".
        custom_prompt: Custom prompt (required when report_format is "custom").
        extra_instructions: Additional instructions appended to the template prompt.
        language: Language code (default: "en").

    Returns the generation status.
    """
    from notebooklm import ReportFormat

    client = await get_client()

    fmt_map = {
        "briefing_doc": ReportFormat.BRIEFING_DOC,
        "study_guide": ReportFormat.STUDY_GUIDE,
        "blog_post": ReportFormat.BLOG_POST,
        "custom": ReportFormat.CUSTOM,
    }
    fmt = fmt_map.get(report_format.lower(), ReportFormat.BRIEFING_DOC)

    status = await client.artifacts.generate_report(
        notebook_id,
        report_format=fmt,
        language=language,
        custom_prompt=custom_prompt,
        extra_instructions=extra_instructions,
    )
    return json.dumps(_serialize(status), indent=2)


@mcp.tool()
async def generate_quiz(
    notebook_id: str,
    instructions: str | None = None,
    quantity: str | None = None,
    difficulty: str | None = None,
) -> str:
    """Generate a quiz from notebook sources.

    Args:
        notebook_id: The notebook ID.
        instructions: Custom instructions for quiz generation.
        quantity: Number of questions — "less", "default", or "more".
        difficulty: Difficulty level — "easy", "medium", or "hard".

    Returns the generation status.
    """
    from notebooklm import QuizDifficulty, QuizQuantity

    client = await get_client()

    qty = None
    if quantity:
        qty_map = {
            "less": QuizQuantity.LESS,
            "default": QuizQuantity.DEFAULT,
            "more": QuizQuantity.MORE,
        }
        qty = qty_map.get(quantity.lower())

    diff = None
    if difficulty:
        diff_map = {
            "easy": QuizDifficulty.EASY,
            "medium": QuizDifficulty.MEDIUM,
            "hard": QuizDifficulty.HARD,
        }
        diff = diff_map.get(difficulty.lower())

    status = await client.artifacts.generate_quiz(
        notebook_id,
        instructions=instructions,
        quantity=qty,
        difficulty=diff,
    )
    return json.dumps(_serialize(status), indent=2)


@mcp.tool()
async def generate_flashcards(
    notebook_id: str,
    instructions: str | None = None,
    quantity: str | None = None,
    difficulty: str | None = None,
) -> str:
    """Generate flashcards from notebook sources.

    Args:
        notebook_id: The notebook ID.
        instructions: Custom instructions for flashcard generation.
        quantity: Number of cards — "less", "default", or "more".
        difficulty: Difficulty level — "easy", "medium", or "hard".

    Returns the generation status.
    """
    from notebooklm import QuizDifficulty, QuizQuantity

    client = await get_client()

    qty = None
    if quantity:
        qty_map = {
            "less": QuizQuantity.LESS,
            "default": QuizQuantity.DEFAULT,
            "more": QuizQuantity.MORE,
        }
        qty = qty_map.get(quantity.lower())

    diff = None
    if difficulty:
        diff_map = {
            "easy": QuizDifficulty.EASY,
            "medium": QuizDifficulty.MEDIUM,
            "hard": QuizDifficulty.HARD,
        }
        diff = diff_map.get(difficulty.lower())

    status = await client.artifacts.generate_flashcards(
        notebook_id,
        instructions=instructions,
        quantity=qty,
        difficulty=diff,
    )
    return json.dumps(_serialize(status), indent=2)


@mcp.tool()
async def generate_infographic(
    notebook_id: str,
    instructions: str | None = None,
    orientation: str | None = None,
    detail_level: str | None = None,
    language: str = "en",
    wait: bool = True,
) -> str:
    """Generate an infographic from notebook sources.

    Args:
        notebook_id: The notebook ID.
        instructions: Custom instructions for the infographic.
        orientation: Layout — "landscape", "portrait", or "square".
        detail_level: Detail — "low", "medium", or "high".
        language: Language code (default: "en").
        wait: If True (default), wait for generation to complete.

    Returns the generation status.
    """
    from notebooklm import InfographicDetail, InfographicOrientation

    client = await get_client()

    orient = None
    if orientation:
        orient_map = {
            "landscape": InfographicOrientation.LANDSCAPE,
            "portrait": InfographicOrientation.PORTRAIT,
            "square": InfographicOrientation.SQUARE,
        }
        orient = orient_map.get(orientation.lower())

    detail = None
    if detail_level:
        detail_map = {
            "low": InfographicDetail.LOW,
            "medium": InfographicDetail.MEDIUM,
            "high": InfographicDetail.HIGH,
        }
        detail = detail_map.get(detail_level.lower())

    status = await client.artifacts.generate_infographic(
        notebook_id,
        language=language,
        instructions=instructions,
        orientation=orient,
        detail_level=detail,
    )

    if wait and status.task_id:
        await client.artifacts.wait_for_completion(notebook_id, status.task_id)

    return json.dumps(_serialize(status), indent=2)


@mcp.tool()
async def generate_slide_deck(
    notebook_id: str,
    instructions: str | None = None,
    slide_format: str | None = None,
    slide_length: str | None = None,
    language: str = "en",
    wait: bool = True,
) -> str:
    """Generate a slide deck (presentation) from notebook sources.

    Args:
        notebook_id: The notebook ID.
        instructions: Custom instructions for the slide deck.
        slide_format: Format — "detailed" or "presenter".
        slide_length: Length — "short", "medium", or "long".
        language: Language code (default: "en").
        wait: If True (default), wait for generation to complete.

    Returns the generation status.
    """
    from notebooklm import SlideDeckFormat, SlideDeckLength

    client = await get_client()

    fmt = None
    if slide_format:
        fmt_map = {
            "detailed": SlideDeckFormat.DETAILED,
            "presenter": SlideDeckFormat.PRESENTER,
        }
        fmt = fmt_map.get(slide_format.lower())

    length = None
    if slide_length:
        len_map = {
            "short": SlideDeckLength.SHORT,
            "medium": SlideDeckLength.MEDIUM,
            "long": SlideDeckLength.LONG,
        }
        length = len_map.get(slide_length.lower())

    status = await client.artifacts.generate_slide_deck(
        notebook_id,
        language=language,
        instructions=instructions,
        slide_format=fmt,
        slide_length=length,
    )

    if wait and status.task_id:
        await client.artifacts.wait_for_completion(notebook_id, status.task_id)

    return json.dumps(_serialize(status), indent=2)


@mcp.tool()
async def generate_data_table(
    notebook_id: str,
    instructions: str | None = None,
    language: str = "en",
) -> str:
    """Generate a data table from notebook sources.

    Args:
        notebook_id: The notebook ID.
        instructions: Custom instructions describing the table structure.
        language: Language code (default: "en").

    Returns the generation status.
    """
    client = await get_client()
    status = await client.artifacts.generate_data_table(
        notebook_id,
        language=language,
        instructions=instructions,
    )
    return json.dumps(_serialize(status), indent=2)


@mcp.tool()
async def generate_mind_map(notebook_id: str) -> str:
    """Generate an interactive mind map from notebook sources.

    Args:
        notebook_id: The notebook ID.

    Returns the mind map JSON data and note_id.
    """
    client = await get_client()
    result = await client.artifacts.generate_mind_map(notebook_id)
    return json.dumps(_serialize(result), indent=2)


@mcp.tool()
async def download_audio(
    notebook_id: str, output_path: str, artifact_id: str | None = None
) -> str:
    """Download an Audio Overview to a local file.

    Args:
        notebook_id: The notebook ID.
        output_path: Local file path to save the audio (e.g., "podcast.mp3").
        artifact_id: Specific artifact ID. If None, downloads the first completed audio.

    Returns the output file path.
    """
    safe_path = _sanitize_output_path(output_path)
    client = await get_client()
    path = await client.artifacts.download_audio(notebook_id, str(safe_path), artifact_id=artifact_id)
    return json.dumps({"downloaded": str(path)})


@mcp.tool()
async def download_video(
    notebook_id: str, output_path: str, artifact_id: str | None = None
) -> str:
    """Download a Video Overview to a local file.

    Args:
        notebook_id: The notebook ID.
        output_path: Local file path to save the video (e.g., "overview.mp4").
        artifact_id: Specific artifact ID. If None, downloads the first completed video.

    Returns the output file path.
    """
    safe_path = _sanitize_output_path(output_path)
    client = await get_client()
    path = await client.artifacts.download_video(notebook_id, str(safe_path), artifact_id=artifact_id)
    return json.dumps({"downloaded": str(path)})


@mcp.tool()
async def download_quiz(
    notebook_id: str,
    output_path: str,
    output_format: str = "json",
    artifact_id: str | None = None,
) -> str:
    """Download a quiz to a local file.

    Args:
        notebook_id: The notebook ID.
        output_path: Local file path to save the quiz.
        output_format: Format — "json", "markdown", or "html".
        artifact_id: Specific artifact ID. If None, downloads the first quiz.

    Returns the output file path.
    """
    safe_path = _sanitize_output_path(output_path)
    client = await get_client()
    path = await client.artifacts.download_quiz(
        notebook_id, str(safe_path), output_format=output_format, artifact_id=artifact_id
    )
    return json.dumps({"downloaded": str(path)})


@mcp.tool()
async def download_flashcards(
    notebook_id: str,
    output_path: str,
    output_format: str = "json",
    artifact_id: str | None = None,
) -> str:
    """Download flashcards to a local file.

    Args:
        notebook_id: The notebook ID.
        output_path: Local file path to save the flashcards.
        output_format: Format — "json", "markdown", or "html".
        artifact_id: Specific artifact ID. If None, downloads the first flashcards.

    Returns the output file path.
    """
    safe_path = _sanitize_output_path(output_path)
    client = await get_client()
    path = await client.artifacts.download_flashcards(
        notebook_id, str(safe_path), output_format=output_format, artifact_id=artifact_id
    )
    return json.dumps({"downloaded": str(path)})


@mcp.tool()
async def download_slide_deck(
    notebook_id: str, output_path: str, artifact_id: str | None = None
) -> str:
    """Download a slide deck to a local file (PDF or PPTX).

    Args:
        notebook_id: The notebook ID.
        output_path: Local file path to save the slides (e.g., "slides.pdf" or "slides.pptx").
        artifact_id: Specific artifact ID. If None, downloads the first slide deck.

    Returns the output file path.
    """
    safe_path = _sanitize_output_path(output_path)
    client = await get_client()
    path = await client.artifacts.download_slide_deck(
        notebook_id, str(safe_path), artifact_id=artifact_id
    )
    return json.dumps({"downloaded": str(path)})


@mcp.tool()
async def download_infographic(
    notebook_id: str, output_path: str, artifact_id: str | None = None
) -> str:
    """Download an infographic to a local file (PNG).

    Args:
        notebook_id: The notebook ID.
        output_path: Local file path to save the image (e.g., "infographic.png").
        artifact_id: Specific artifact ID. If None, downloads the first infographic.

    Returns the output file path.
    """
    safe_path = _sanitize_output_path(output_path)
    client = await get_client()
    path = await client.artifacts.download_infographic(
        notebook_id, str(safe_path), artifact_id=artifact_id
    )
    return json.dumps({"downloaded": str(path)})


@mcp.tool()
async def download_mind_map(
    notebook_id: str, output_path: str
) -> str:
    """Download a mind map as JSON.

    Args:
        notebook_id: The notebook ID.
        output_path: Local file path to save the JSON (e.g., "mindmap.json").

    Returns the output file path.
    """
    safe_path = _sanitize_output_path(output_path)
    client = await get_client()
    path = await client.artifacts.download_mind_map(notebook_id, str(safe_path))
    return json.dumps({"downloaded": str(path)})


@mcp.tool()
async def download_data_table(
    notebook_id: str, output_path: str, artifact_id: str | None = None
) -> str:
    """Download a data table as CSV.

    Args:
        notebook_id: The notebook ID.
        output_path: Local file path to save the CSV (e.g., "data.csv").
        artifact_id: Specific artifact ID. If None, downloads the first data table.

    Returns the output file path.
    """
    safe_path = _sanitize_output_path(output_path)
    client = await get_client()
    path = await client.artifacts.download_data_table(
        notebook_id, str(safe_path), artifact_id=artifact_id
    )
    return json.dumps({"downloaded": str(path)})


# ═══════════════════════════════════════════════════════════════════════════════
#  RESEARCH TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def start_research(
    notebook_id: str,
    query: str,
    source: str = "web",
    mode: str = "fast",
) -> str:
    """Start a research session to find and import sources.

    Args:
        notebook_id: The notebook ID.
        query: The research query.
        source: Search source — "web" or "drive".
        mode: Search mode — "fast" or "deep" (deep only for web).

    Returns the research task ID and initial status.
    """
    client = await get_client()
    result = await client.research.start(notebook_id, query, source=source, mode=mode)
    return json.dumps(_serialize(result), indent=2)


@mcp.tool()
async def poll_research(notebook_id: str) -> str:
    """Poll for research results.

    Args:
        notebook_id: The notebook ID.

    Returns the current research status with discovered sources and summary.
    """
    client = await get_client()
    result = await client.research.poll(notebook_id)
    return json.dumps(_serialize(result), indent=2)


@mcp.tool()
async def import_research_sources(
    notebook_id: str,
    task_id: str,
    sources: list[dict[str, str]],
) -> str:
    """Import discovered research sources into the notebook.

    Args:
        notebook_id: The notebook ID.
        task_id: The research task ID from start_research.
        sources: List of sources to import, each with 'url' and 'title' keys.

    Returns the import result.
    """
    client = await get_client()
    result = await client.research.import_sources(notebook_id, task_id, sources)
    return json.dumps(_serialize(result), indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARING TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def share_notebook(notebook_id: str, public: bool = True) -> str:
    """Toggle notebook sharing (public/private).

    Args:
        notebook_id: The notebook ID.
        public: True to make public, False to make private.

    Returns sharing status with URL.
    """
    client = await get_client()
    result = await client.notebooks.share(notebook_id, public=public)
    return json.dumps(_serialize(result), indent=2)


@mcp.tool()
async def get_share_url(notebook_id: str) -> str:
    """Get the share URL for a notebook.

    Args:
        notebook_id: The notebook ID.

    Returns the share URL string.
    """
    client = await get_client()
    url = client.notebooks.get_share_url(notebook_id)
    return json.dumps({"url": url})


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    mcp.run(transport="stdio")
