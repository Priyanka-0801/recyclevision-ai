def compute_batch_summary(detection_result: dict, classification: dict, color_summary: dict) -> dict:
    """
    Aggregate all signals into a final batch report.

    Args:
        detection_result: output from detector.detect_bottles()
        classification: output from classifier.classify_batch_with_vlm()
        color_summary: aggregated color distribution across all bottles

    Returns:
        Full batch report dict
    """
    bottle_count = detection_result["bottle_count"]
    detections = detection_result["detections"]

    # Confidence stats from YOLO
    if detections:
        confidences = [d["confidence"] for d in detections]
        avg_confidence = round(sum(confidences) / len(confidences), 3)
        low_conf_count = sum(1 for d in detections if d["needs_review"])
    else:
        avg_confidence = 0.0
        low_conf_count = 0

    # Estimated weight and value
    avg_bottle_weight_kg = 0.025  # ~25g per empty PET bottle
    estimated_weight_kg = round(bottle_count * avg_bottle_weight_kg, 2)
    value_per_kg = classification.get("estimated_value_per_kg", 0.15)
    estimated_value_usd = round(estimated_weight_kg * value_per_kg, 2)

    # Human review flag — trigger if any of these conditions are met
    needs_review = (
        classification.get("needs_human_review", False)
        or classification.get("confidence", 1.0) < 0.5
        or classification.get("contamination_risk") == "high"
        or low_conf_count > bottle_count * 0.3  # >30% low-confidence detections
    )

    review_reasons = []
    if classification.get("needs_human_review"):
        review_reasons.append(classification.get("review_reason", "VLM flagged"))
    if classification.get("confidence", 1.0) < 0.5:
        review_reasons.append("Low VLM confidence")
    if classification.get("contamination_risk") == "high":
        review_reasons.append("High contamination risk detected")
    if low_conf_count > bottle_count * 0.3:
        review_reasons.append(f"{low_conf_count} bottles had low detection confidence")

    return {
        "bottle_count": bottle_count,
        "avg_detection_confidence": avg_confidence,
        "low_confidence_detections": low_conf_count,
        "color_distribution": color_summary,
        "pet_percentage": classification.get("pet_percentage", 0),
        "non_pet_percentage": classification.get("non_pet_percentage", 0),
        "contamination_risk": classification.get("contamination_risk", "unknown"),
        "contamination_notes": classification.get("contamination_notes", ""),
        "batch_grade": classification.get("batch_grade", "D"),
        "grade_rationale": classification.get("grade_rationale", ""),
        "estimated_weight_kg": estimated_weight_kg,
        "estimated_value_usd": estimated_value_usd,
        "vlm_confidence": classification.get("confidence", 0.0),
        "needs_human_review": needs_review,
        "review_reasons": review_reasons
    }
