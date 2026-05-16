# ♻️ RecycleVision AI — Prototype

A vision-based AI system for recycling facility batch analysis. Upload an image of incoming plastic bottles and get instant operational insights: bottle count, PET composition, color distribution, contamination risk, batch grade, and estimated value.

---

## Quick Start (Google Colab)

Run the following cells in a Colab notebook:

```python
# Cell 1 — Install dependencies
!pip install ultralytics streamlit anthropic opencv-python-headless pillow -q

# Cell 2 — Clone or upload project files
# Upload detector.py, classifier.py, grader.py, app.py to Colab

# Cell 3 — Run Streamlit with tunnel
!pip install pyngrok -q
from pyngrok import ngrok
import subprocess, time

proc = subprocess.Popen(["streamlit", "run", "app.py", "--server.port=8501"])
time.sleep(3)
public_url = ngrok.connect(8501)
print("App running at:", public_url)
```

Then open the printed URL in your browser.

---

## What It Does

| Output | Method | Notes |
|--------|--------|-------|
| Bottle count | YOLOv8n (COCO) | Detects class `bottle` (id=39) |
| Confidence per detection | YOLOv8n | Built-in confidence scores |
| Color distribution | OpenCV (HSV) | Per-bottle crop analysis |
| PET vs non-PET % | VLM (Vision-Language Model) | Prototype approach |
| Contamination risk | VLM | Low / Medium / High |
| Batch grade (A–D) | VLM | Based on composition + contamination |
| Estimated value (USD) | Rule-based formula | Weight × value/kg |
| Human review flag | Multi-signal logic | Triggers on low confidence or high contamination |

---

## Architecture

```
Image Input
    │
    ▼
┌─────────────────────┐
│  YOLOv8n (COCO)     │  ──▶ Bottle count, bounding boxes, confidence scores
└─────────────────────┘
    │ detected crops
    ▼
┌─────────────────────┐
│  OpenCV (HSV)       │  ──▶ Color distribution per bottle (clear/blue/green/brown)
└─────────────────────┘
    │ full image + context
    ▼
┌─────────────────────┐
│  VLM (via API)      │  ──▶ PET %, contamination risk, batch grade, value/kg
└─────────────────────┘
    │ all signals
    ▼
┌─────────────────────┐
│  Grader (logic)     │  ──▶ Final report: value estimate, review flag, JSON export
└─────────────────────┘
    │
    ▼
Streamlit Dashboard
```

---

## AI/ML Approach — Choices and Trade-offs

### What I chose

**YOLOv8n for detection**
- Pre-trained on COCO dataset which includes a `bottle` class
- Fast inference even on CPU — suitable for Colab and initial deployment
- Built-in confidence scores per detection, directly usable as a review signal
- Easily fine-tunable on custom recycling data without architectural changes

**OpenCV for color analysis**
- Zero inference cost, runs instantly on cropped bottle regions
- HSV color space provides better color separation than RGB
- Clear/transparent bottles are a strong heuristic for PET material

**VLM for PET classification and batch grading**
- Eliminates need for labeled training data at prototype stage
- Handles multi-task output (classification + grading + contamination) in one call
- Flexible reasoning about ambiguous cases
- Trade-off: API cost (~$0.002/image), latency (~5s), not suitable for real-time conveyor belt use

### What I rejected

**Fine-tuned classifier (e.g. EfficientNet, ResNet)**
- Best long-term solution for PET vs non-PET
- Rejected for prototype due to lack of labeled data and training time
- This is the clear production migration path

**Segmentation models (e.g. SAM, YOLOv8-seg)**
- Better for precise material boundary detection
- Overkill for batch-level composition estimation
- Would add value for per-bottle contamination detection in future

**OCR for resin codes**
- Recycling resin codes (1=PET, 2=HDPE etc.) are the ground truth signal
- In practice, codes are often obscured, dirty, or facing down
- Worth adding as a secondary signal in production

---

## Data Strategy

### Prototype (now)
- YOLOv8 uses COCO pre-training — no custom data needed
- VLM handles classification without training data

### Short-term (1–3 months)
- Label 500–1000 bottle images with: material type, color, contamination
- Use Label Studio or Roboflow for annotation
- Fine-tune a binary classifier (PET / non-PET) on top of YOLOv8 crops

### Long-term (3–12 months)
- Deploy model at facility, collect predictions + operator corrections
- Active learning: flag uncertain predictions for human labeling
- Resin code OCR as ground truth signal when visible
- Expand to multi-class: PET, HDPE, PVC, LDPE, PP, PS

---

## Evaluation Plan

Accuracy alone is insufficient for this business. Key metrics:

| Metric | Why it matters |
|--------|---------------|
| **Precision on PET** | False positives send non-PET to PET stream → rejected loads |
| **Recall on contamination** | Missing contamination → downstream damage to recycling equipment |
| **False review rate** | Too many human reviews → operational bottleneck |
| **Throughput (images/sec)** | Must keep up with conveyor belt speed |
| **Value estimation error (%)** | Directly impacts financial decisions |
| **Operator override rate** | High rate = model not trusted = system unused |

---

## MLOps — Prototype to Production

| Stage | Approach |
|-------|----------|
| **Serving** | Replace VLM with on-premise fine-tuned classifier for latency + cost |
| **Hardware** | NVIDIA edge GPU (Jetson Orin) for real-time conveyor belt inference |
| **Monitoring** | Log predictions + confidence scores; alert on distribution shift |
| **Retraining** | Operator corrections feed back as labeled data; retrain monthly |
| **Versioning** | MLflow or Weights & Biases for experiment tracking |
| **Deployment** | Docker container, blue-green deployment to avoid downtime |
| **Fallback** | If model confidence < threshold, always route to human review |

---

## Assumptions

- Input is a single overhead image of a batch (not real-time video stream)
- Average empty PET bottle weight assumed at 25g for value estimation
- PET market price assumed at $0.10–$0.30/kg (varies by region and market)
- "Bottle" class in COCO is sufficient for initial detection without fine-tuning
- VLM has no access to facility-specific historical data

## Known Limitations

- PET classification relies on visual appearance, not chemical composition
- YOLOv8 COCO weights may miss bottles that are heavily occluded or stacked
- VLM latency (~5s) is not suitable for real-time conveyor belt use
- Value estimation is a rough approximation, not a market-connected price feed
- Color-based PET heuristic (clear = likely PET) is imprecise for mixed batches

---

## Sample Output (JSON)

```json
{
  "bottle_count": 12,
  "avg_detection_confidence": 0.74,
  "low_confidence_detections": 2,
  "color_distribution": {"clear": 0.52, "blue": 0.21, "green": 0.10, "brown": 0.07, "other": 0.10},
  "pet_percentage": 68,
  "non_pet_percentage": 32,
  "contamination_risk": "medium",
  "contamination_notes": "Some caps and labels present; minor food residue visible",
  "batch_grade": "B",
  "grade_rationale": "Majority PET with moderate contamination, acceptable for standard recycling",
  "estimated_weight_kg": 0.3,
  "estimated_value_usd": 0.05,
  "vlm_confidence": 0.81,
  "needs_human_review": false,
  "review_reasons": []
}
```




## Demo Video
https://drive.google.com/file/d/1GdjqYa3K1rykFb_57mtt2wKFXz0zBx8-/view?usp=sharing
