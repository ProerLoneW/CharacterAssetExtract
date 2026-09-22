"""Multi-reference image restoration tools backed by UniAPI."""

from .uniapi_gemini_image_client import (
    GEMINI_IMAGE_MODEL_PROFILES,
    GeminiRestorationResult,
    UniAPIGeminiImageClient,
    resolve_model_image_size,
    restore_gemini_reference,
)

from .uniapi_image_client import (
    OPENAI_IMAGE_MODEL_PROFILES,
    RestorationResult,
    UniAPIImageClient,
    UniAPIImageError,
    resolve_openai_model_size,
    restore_reference,
)

from .extract_character_clothing import (
    ClothingExtractionPlan,
    ClothingExtractionResult,
    extract_character_combined,
    extract_character_pants,
    extract_character_top,
    plan_clothing_extraction,
)

from .extract_character_clothing_openai import (
    DEFAULT_OPENAI_IMAGE_MODEL,
    SUPPORTED_OPENAI_MODELS,
    extract_character_combined_openai,
    extract_character_pants_openai,
    extract_character_top_openai,
    execute_plan_openai,
)

from .two_stage_clothing_pipeline import (
    ALL_MODELS,
    TwoStagePlan,
    plan_two_stage,
    run_two_stage,
)

__all__ = [
    "ALL_MODELS",
    "ClothingExtractionPlan",
    "ClothingExtractionResult",
    "DEFAULT_OPENAI_IMAGE_MODEL",
    "GEMINI_IMAGE_MODEL_PROFILES",
    "GeminiRestorationResult",
    "OPENAI_IMAGE_MODEL_PROFILES",
    "RestorationResult",
    "SUPPORTED_OPENAI_MODELS",
    "TwoStagePlan",
    "UniAPIGeminiImageClient",
    "UniAPIImageClient",
    "UniAPIImageError",
    "extract_character_combined",
    "extract_character_combined_openai",
    "extract_character_pants",
    "extract_character_pants_openai",
    "extract_character_top",
    "extract_character_top_openai",
    "execute_plan_openai",
    "plan_clothing_extraction",
    "plan_two_stage",
    "resolve_model_image_size",
    "resolve_openai_model_size",
    "restore_gemini_reference",
    "restore_reference",
    "run_two_stage",
]
