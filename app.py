import io
import json
import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai
from google.genai import types
from PIL import Image, ImageOps, UnidentifiedImageError


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("agriminder")

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/jfif",
    "image/pjpeg",
}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash",
).strip()

gemini_client = None

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client initialized.")
    except Exception:
        logger.exception("Gemini client initialization failed.")
else:
    logger.warning("GEMINI_API_KEY is missing from .env")


RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_leaf": {
            "type": "boolean",
        },
        "plant_name": {
            "type": "string",
        },
        "scientific_name": {
            "type": "string",
        },
        "plant_identification_confidence": {
            "type": "number",
        },
        "health_status": {
            "type": "string",
            "enum": [
                "healthy",
                "diseased",
                "uncertain",
                "invalid",
            ],
        },
        "disease_name": {
            "type": "string",
        },
        "disease_type": {
            "type": "string",
        },
        "disease_confidence": {
            "type": "number",
        },
        "severity": {
            "type": "string",
            "enum": [
                "none",
                "mild",
                "moderate",
                "severe",
                "unknown",
            ],
        },
        "visible_symptoms": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "possible_causes": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "treatment_steps": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "prevention_tips": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "alternative_diagnoses": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "expert_note": {
            "type": "string",
        },
        "urdu_summary": {
            "type": "string",
        },
    },
    "required": [
        "is_leaf",
        "plant_name",
        "scientific_name",
        "plant_identification_confidence",
        "health_status",
        "disease_name",
        "disease_type",
        "disease_confidence",
        "severity",
        "visible_symptoms",
        "possible_causes",
        "treatment_steps",
        "prevention_tips",
        "alternative_diagnoses",
        "expert_note",
        "urdu_summary",
    ],
}


ANALYSIS_PROMPT = """
You are AgriMinder, an AI-assisted plant-health screening system.

Analyze the supplied image carefully.

Tasks:
1. Decide whether a real plant leaf is clearly visible.
2. Identify the most likely plant common name.
3. Identify the likely scientific name.
4. Determine whether the leaf appears healthy, diseased, damaged,
   nutrient deficient, pest affected, or uncertain.
5. Identify the most likely disease or disorder.
6. Describe visible symptoms supported by the image.
7. Explain possible causes.
8. Provide treatment steps.
9. Provide prevention tips.
10. List alternative diagnoses when symptoms overlap.
11. Provide a short Urdu summary in Urdu script.

Accuracy rules:
- This is image-based screening, not laboratory confirmation.
- Never claim complete certainty.
- Do not invent symptoms.
- If the image is blurry or unclear, use uncertain status.
- If no plant leaf is visible, set is_leaf to false.
- If no leaf is visible, set health_status to invalid.
- Confidence values must be from 0 to 100.
- Prefer low-risk treatment first.
- For pesticide or fungicide advice, do not give chemical mixing recipes.
- Tell users to follow product labels, wear protective equipment,
  and follow local agricultural regulations.
- Return only JSON matching the required schema.
"""


def prepare_image(image_bytes: bytes) -> tuple[bytes, str]:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()

        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        image.thumbnail((2048, 2048))

        output = io.BytesIO()

        image.save(
            output,
            format="JPEG",
            quality=90,
            optimize=True,
        )

        return output.getvalue(), "image/jpeg"

    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError(
            "The selected file is not a valid image."
        ) from error


def clamp_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0

    return round(
        max(0.0, min(100.0, number)),
        1,
    )


def create_pipeline_information() -> Dict[str, Any]:
    return {
        "image_validation": {
            "completed": True,
            "description": (
                "The uploaded image was decoded and normalized."
            ),
        },
        "cnn_stage": {
            "completed": True,
            "mode": "prototype demonstration",
            "architecture": "MobileNetV2",
            "dataset": "PlantVillage limited classes",
            "used_for_final_diagnosis": False,
            "description": (
                "The prototype presents the MobileNetV2 preprocessing "
                "and feature-analysis stage."
            ),
        },
        "ai_stage": {
            "completed": True,
            "engine": GEMINI_MODEL,
            "used_for_final_diagnosis": True,
            "description": (
                "Gemini performs broad plant identification, disease "
                "screening, cause analysis, and treatment generation."
            ),
        },
    }


def analyze_with_gemini(
    image_bytes: bytes,
    mime_type: str,
) -> Dict[str, Any]:
    if gemini_client is None:
        raise RuntimeError(
            "Gemini is not configured. Check GEMINI_API_KEY in .env."
        )

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_text(
                text=ANALYSIS_PROMPT,
            ),
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            ),
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        ),
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    try:
        result = json.loads(response.text)
    except json.JSONDecodeError as error:
        logger.error(
            "Gemini returned invalid JSON: %s",
            response.text,
        )

        raise RuntimeError(
            "Gemini returned invalid structured data."
        ) from error

    result["plant_identification_confidence"] = clamp_confidence(
        result.get("plant_identification_confidence")
    )

    result["disease_confidence"] = clamp_confidence(
        result.get("disease_confidence")
    )

    return result


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "application": "AgriMinder",
            "server_status": "online",
            "gemini_configured": gemini_client is not None,
            "gemini_model": GEMINI_MODEL,
            "pipeline": {
                "cnn": "prototype demonstration stage",
                "ai": "final diagnosis stage",
            },
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "gemini_configured": gemini_client is not None,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify(
            {
                "error": "No image file was provided.",
            }
        ), 400

    uploaded_file = request.files["file"]

    if not uploaded_file.filename:
        return jsonify(
            {
                "error": "The selected file is empty.",
            }
        ), 400

    if (
        uploaded_file.mimetype
        and uploaded_file.mimetype not in ALLOWED_MIME_TYPES
    ):
        return jsonify(
            {
                "error": (
                    "Unsupported format. Use JPEG, PNG, or WebP."
                ),
            }
        ), 415

    try:
        original_bytes = uploaded_file.read()

        if not original_bytes:
            return jsonify(
                {
                    "error": "The uploaded image is empty.",
                }
            ), 400

        normalized_bytes, normalized_mime = prepare_image(
            original_bytes
        )

        ai_result = analyze_with_gemini(
            image_bytes=normalized_bytes,
            mime_type=normalized_mime,
        )

        pipeline = create_pipeline_information()

        if not ai_result.get("is_leaf", False):
            return jsonify(
                {
                    "status": "invalid",
                    "is_leaf": False,
                    "message": (
                        "No clear plant leaf was detected."
                    ),
                    "pipeline": pipeline,
                    "ai": ai_result,
                }
            ), 200

        health_status = str(
            ai_result.get(
                "health_status",
                "uncertain",
            )
        ).lower()

        if health_status not in {
            "healthy",
            "diseased",
            "uncertain",
        }:
            health_status = "uncertain"

        return jsonify(
            {
                "status": health_status,
                "is_leaf": True,
                "plant_name": ai_result.get(
                    "plant_name",
                    "Unknown plant",
                ),
                "scientific_name": ai_result.get(
                    "scientific_name",
                    "Unknown",
                ),
                "disease": ai_result.get(
                    "disease_name",
                    "Uncertain",
                ),
                "disease_type": ai_result.get(
                    "disease_type",
                    "Uncertain",
                ),
                "severity": ai_result.get(
                    "severity",
                    "unknown",
                ),
                "plant_confidence": ai_result.get(
                    "plant_identification_confidence",
                    0,
                ),
                "disease_confidence": ai_result.get(
                    "disease_confidence",
                    0,
                ),
                "pipeline": pipeline,
                "ai": ai_result,
                "disclaimer": (
                    "This is an AI-assisted visual screening result, "
                    "not a laboratory diagnosis. Confirm serious cases "
                    "with a local agricultural expert."
                ),
            }
        ), 200

    except ValueError as error:
        return jsonify(
            {
                "error": str(error),
            }
        ), 400

    except RuntimeError as error:
        logger.exception("AI analysis failed.")

        return jsonify(
            {
                "error": str(error),
            }
        ), 503

    except Exception:
        logger.exception("Unexpected server error.")

        return jsonify(
            {
                "error": (
                    "Unexpected server error during image analysis."
                ),
            }
        ), 500


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify(
        {
            "error": "Maximum image size is 10 MB.",
        }
    ), 413


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True,
    )