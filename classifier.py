import anthropic
import base64
import json
import numpy as np
from PIL import Image
import io

def numpy_to_base64(image: np.ndarray) -> str:
    """Convert numpy image array to base64 string for API."""
    pil_img = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=85)
    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

def classify_batch_with_vlm(image: np.ndarray, bottle_count: int, color_summary: dict, api_key: str) -> dict:
    """
    Use a Vision-Language Model to classify PET vs non-PET composition,
    assess contamination risk, and estimate batch grade.

    This is the prototype approach. In production, this would be replaced
    by a fine-tuned image classifier (e.g., EfficientNet or ResNet)
    trained on labeled recycling facility data.

    Args:
        image: full scene image as numpy array (RGB)
        bottle_count: number of bottles detected by YOLO
        color_summary: color distribution from OpenCV analysis
        api_key: Anthropic API key

    Returns:
        dict with classification results
    """
    client = anthropic.Anthropic(api_key=api_key)

    image_b64 = numpy_to_base64(image)

    prompt = f"""You are an expert recycling facility analyst. Analyze this image of plastic bottles.

Context from computer vision analysis:
- Detected bottle count: {bottle_count}
- Color distribution: {json.dumps(color_summary)}

Please analyze the image and return ONLY a valid JSON object with these exact keys:
{{
  "pet_percentage": <integer 0-100, estimated % of PET plastic bottles>,
  "non_pet_percentage": <integer 0-100, estimated % of non-PET plastic>,
  "contamination_risk": <"low", "medium", or "high">,
  "contamination_notes": <brief string explaining contamination observations>,
  "batch_grade": <"A", "B", "C", or "D">,
  "grade_rationale": <one sentence explaining the grade>,
  "estimated_value_per_kg": <float in USD, typical PET is $0.10-$0.30/kg>,
  "confidence": <float 0.0-1.0, your confidence in this assessment>,
  "needs_human_review": <true or false>,
  "review_reason": <string, reason for human review or "none">
}}

Grading scale:
- A: >80% PET, low contamination, high value
- B: 60-80% PET, moderate contamination
- C: 40-60% PET or notable contamination
- D: <40% PET or high contamination risk

Return ONLY the JSON object, no explanation, no markdown."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    result = json.loads(raw)
    return result


def get_fallback_classification(color_summary: dict) -> dict:
    """
    Rule-based fallback if VLM API call fails.
    Uses color heuristics: clear bottles are likely PET.
    This is intentionally simple and would not be used in production.
    """
    clear_ratio = color_summary.get("clear", 0)
    blue_ratio = color_summary.get("blue", 0)

    pet_estimate = int((clear_ratio + blue_ratio * 0.7) * 100)
    pet_estimate = min(max(pet_estimate, 10), 90)  # clamp

    grade = "A" if pet_estimate > 75 else "B" if pet_estimate > 55 else "C" if pet_estimate > 35 else "D"

    return {
        "pet_percentage": pet_estimate,
        "non_pet_percentage": 100 - pet_estimate,
        "contamination_risk": "medium",
        "contamination_notes": "Rule-based fallback — VLM unavailable",
        "batch_grade": grade,
        "grade_rationale": "Estimated from color analysis only (fallback mode)",
        "estimated_value_per_kg": round(0.10 + (pet_estimate / 100) * 0.20, 2),
        "confidence": 0.35,
        "needs_human_review": True,
        "review_reason": "VLM classification unavailable; rule-based fallback used"
    }
