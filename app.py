import streamlit as st
import numpy as np
from PIL import Image
import cv2
import json

from detector import load_model, detect_bottles, analyze_color
from classifier import classify_batch_with_vlm, get_fallback_classification
from grader import compute_batch_summary

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RecycleVision AI",
    page_icon="♻️",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .grade-A { color: #00c853; font-size: 3rem; font-weight: bold; }
    .grade-B { color: #64dd17; font-size: 3rem; font-weight: bold; }
    .grade-C { color: #ffab00; font-size: 3rem; font-weight: bold; }
    .grade-D { color: #d50000; font-size: 3rem; font-weight: bold; }
    .review-flag {
        background: #fff3e0;
        border-left: 4px solid #ff6f00;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .safe-flag {
        background: #e8f5e9;
        border-left: 4px solid #2e7d32;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("♻️ RecycleVision AI")
st.caption("Vision-based batch analysis for plastic recycling facilities")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Anthropic API Key", type="password", help="Required for VLM-based PET classification")
    conf_threshold = st.slider("Detection Confidence Threshold", 0.1, 0.9, 0.3, 0.05,
                               help="Lower = detect more bottles, but more false positives")
    st.divider()
    st.markdown("**Architecture**")
    st.markdown("""
- 🔍 **YOLOv8n** — bottle detection & count  
- 🎨 **OpenCV** — color distribution  
- 🧠 **VLM** — PET classification & grading  
- 📊 **Rule-based** — value estimation
    """)
    st.caption("Prototype: VLM used for classification. Production would use a fine-tuned EfficientNet classifier.")

# ── File Upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload an image of the recycling batch",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload a photo of plastic bottles for analysis"
)

if uploaded is None:
    st.info("👆 Upload a batch image to get started.")
    st.stop()

# ── Load image ────────────────────────────────────────────────────────────────
pil_image = Image.open(uploaded).convert("RGB")
image_np = np.array(pil_image)

col_img, col_info = st.columns([1.2, 1])

with col_img:
    st.image(pil_image, caption="Uploaded batch image", use_container_width=True)

with col_info:
    st.markdown("### Image Info")
    st.write(f"**Dimensions:** {pil_image.width} × {pil_image.height} px")
    st.write(f"**File size:** {uploaded.size / 1024:.1f} KB")
    st.write(f"**Format:** {pil_image.format or uploaded.type}")

st.divider()

# ── Run Analysis ──────────────────────────────────────────────────────────────
if st.button("🔍 Analyze Batch", type="primary", use_container_width=True):

    with st.spinner("Loading YOLOv8 model..."):
        model = load_model()

    with st.spinner("Detecting bottles..."):
        detection = detect_bottles(image_np, model, conf_threshold=conf_threshold)

    # Color analysis across all detected crops
    all_colors = {"clear": 0, "blue": 0, "green": 0, "brown": 0, "other": 0}
    for det in detection["detections"]:
        crop = det["crop"]
        if crop is not None and crop.size > 0:
            c = analyze_color(crop)
            for k in all_colors:
                all_colors[k] += c["distribution"].get(k, 0)

    n = len(detection["detections"])
    if n > 0:
        color_summary = {k: round(v / n, 2) for k, v in all_colors.items()}
    else:
        color_summary = all_colors

    # VLM Classification
    with st.spinner("Running VLM classification (this may take 5-10 seconds)..."):
        if api_key:
            try:
                classification = classify_batch_with_vlm(image_np, detection["bottle_count"], color_summary, api_key)
            except Exception as e:
                st.warning(f"VLM call failed ({e}). Using rule-based fallback.")
                classification = get_fallback_classification(color_summary)
        else:
            st.warning("No API key provided — using rule-based fallback for classification.")
            classification = get_fallback_classification(color_summary)

    # Final batch summary
    summary = compute_batch_summary(detection, classification, color_summary)

    # ── Results ───────────────────────────────────────────────────────────────
    st.success("✅ Analysis complete!")
    st.divider()

    # Annotated image
    st.subheader("🔍 Detection Output")
    st.image(detection["annotated_image"], caption=f"YOLOv8 detected {summary['bottle_count']} bottle(s)", use_container_width=True)

    st.divider()

    # Key metrics
    st.subheader("📊 Batch Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.metric("🍶 Bottle Count", summary["bottle_count"])
    with m2:
        st.metric("♻️ PET %", f"{summary['pet_percentage']}%")
    with m3:
        st.metric("⚠️ Contamination", summary["contamination_risk"].capitalize())
    with m4:
        st.metric("💰 Est. Value", f"${summary['estimated_value_usd']}")
    with m5:
        st.metric("🧠 VLM Confidence", f"{int(summary['vlm_confidence'] * 100)}%")

    st.divider()

    # Batch Grade
    col_grade, col_details = st.columns([1, 2])

    with col_grade:
        st.subheader("Batch Grade")
        grade = summary["batch_grade"]
        st.markdown(f'<div class="grade-{grade}">{grade}</div>', unsafe_allow_html=True)
        st.caption(summary["grade_rationale"])

    with col_details:
        st.subheader("Composition & Color")
        st.progress(summary["pet_percentage"] / 100, text=f"PET: {summary['pet_percentage']}%")
        st.progress(summary["non_pet_percentage"] / 100, text=f"Non-PET: {summary['non_pet_percentage']}%")
        st.caption("**Color distribution (avg across detected bottles):**")
        for color, val in summary["color_distribution"].items():
            st.progress(val, text=f"{color.capitalize()}: {int(val * 100)}%")

    st.divider()

    # Human review flag
    st.subheader("🚦 Review Flag")
    if summary["needs_human_review"]:
        reasons = "\n".join([f"• {r}" for r in summary["review_reasons"]])
        st.markdown(f'<div class="review-flag">⚠️ <strong>Human review recommended</strong><br>{reasons}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="safe-flag">✅ <strong>No human review needed</strong> — all signals within acceptable thresholds.</div>', unsafe_allow_html=True)

    if summary["contamination_notes"]:
        st.caption(f"📝 Contamination notes: {summary['contamination_notes']}")

    st.divider()

    # Raw JSON export
    with st.expander("📄 Export Full Report (JSON)"):
        # Remove non-serializable items
        export = {k: v for k, v in summary.items() if k != "annotated_image"}
        st.json(export)
        st.download_button(
            "⬇️ Download Report",
            data=json.dumps(export, indent=2),
            file_name="recycling_batch_report.json",
            mime="application/json"
        )
