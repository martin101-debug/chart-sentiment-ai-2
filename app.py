import os
import base64
import json
import re
import io

import streamlit as st
import openai
import cv2
import numpy as np
from PIL import Image

# --- 1. APP HEADER ---
st.set_page_config(page_title="AI Chart Sentiment Scanner", layout="centered")
st.title("📊 Professional Chart AI Sentiment Scanner")
st.write("Snap a photo or upload a screenshot of any trading chart to estimate Bullish vs. Bearish sentiment.")

# --- 2. SECURITY CONFIGURATION ---
st.sidebar.header("🔑 Authentication")

env_api_key = os.getenv("OPENAI_API_KEY")
api_key_input = st.sidebar.text_input(
    "Enter OpenAI API Key (optional if set as environment variable)",
    type="password",
    value="" if env_api_key else "",
)

api_key = api_key_input or env_api_key

if not api_key:
    st.info("Please enter your OpenAI API Key in the sidebar or configure OPENAI_API_KEY in Streamlit Secrets.")
    st.stop()

openai.api_key = api_key

st.sidebar.markdown("### ℹ️ About")
st.sidebar.write("AI-driven visual sentiment scanner for trading charts using multimodal GPT-4o.")

st.sidebar.markdown("### ⚠️ Disclaimer")
st.sidebar.write("This tool is for educational and experimental purposes only and does not constitute financial advice.")

# --- 3. TRADING MODE SELECTOR ---
st.subheader("🎛️ Trading Style Mode")
mode = st.selectbox(
    "Choose analysis mode",
    ["Scalper", "Swing", "Position"],
    index=1,
)

# --- 4. HELPER FUNCTIONS ---

def detect_trendlines(uploaded_file):
    bytes_data = uploaded_file.read()
    uploaded_file.seek(0)

    img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=50,
        maxLineGap=10,
    )

    overlay = cv_img.copy()
    trend_info = []

    if lines is not None:
        for line in lines[:30]:
            x1, y1, x2, y2 = line[0]
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            slope = -(y2 - y1) / (x2 - x1 + 1e-6)
            trend_info.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "slope": slope})

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlay_rgb), trend_info, cv_img


def describe_panels(uploaded_file):
    bytes_data = uploaded_file.read()
    uploaded_file.seek(0)

    img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    h, w, _ = cv_img.shape
    indicator_region = cv_img[int(0.7 * h):, :]

    gray_ind = cv2.cvtColor(indicator_region, cv2.COLOR_BGR2GRAY)
    edges_ind = cv2.Canny(gray_ind, 50, 150)
    edge_density = np.mean(edges_ind > 0)

    if edge_density > 0.02:
        return "Lower panel likely contains oscillating indicators such as RSI or MACD."
    return "No clearly separate indicator panel detected."


def rule_based_sentiment(trend_info):
    if not trend_info:
        return 50.0, 50.0

    avg_slope = float(np.mean([t["slope"] for t in trend_info]))

    if avg_slope > 0.1:
        return 65.0, 35.0
    elif avg_slope < -0.1:
        return 35.0, 65.0
    return 50.0, 50.0


def normalize_to_100(bullish, bearish):
    total = bullish + bearish
    if total == 0:
        return 50.0, 50.0
    bullish = round((bullish / total) * 100, 2)
    bearish = round((bearish / total) * 100, 2)
    return bullish, bearish


def classify_trend(trend_info):
    if not trend_info:
        return "Unclear / Range"

    avg_slope = float(np.mean([t["slope"] for t in trend_info]))

    if avg_slope > 0.2:
        return "Strong Uptrend"
    elif avg_slope > 0.05:
        return "Mild Uptrend"
    elif avg_slope < -0.2:
        return "Strong Downtrend"
    elif avg_slope < -0.05:
        return "Mild Downtrend"
    else:
        return "Sideways / Range"


def compute_risk_score(trend_info, cv_img):
    if cv_img is None:
        return 50

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges > 0)

    line_count = len(trend_info)
    base = 30 + edge_density * 100 + line_count * 2
    return int(max(0, min(100, base)))


def compute_confidence(trend_info, panel_description):
    conf = 40
    if trend_info:
        conf += 20
    if "likely contains oscillating indicators" in panel_description:
        conf += 20
    return int(max(0, min(100, conf)))


def detect_support_resistance(trend_info):
    if not trend_info:
        return "No clear horizontal support/resistance lines detected."

    horizontals = [t for t in trend_info if abs(t["slope"]) < 0.02]
    if len(horizontals) >= 2:
        return "Multiple horizontal lines detected: possible support/resistance zones."
    elif len(horizontals) == 1:
        return "Single horizontal line detected: possible key level."
    else:
        return "Trendlines mostly diagonal: trend-focused, fewer clear horizontal levels."


def candlestick_hint(uploaded_file):
    bytes_data = uploaded_file.read()
    uploaded_file.seek(0)

    img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    vertical_edges = np.sum(edges[:, ::5] > 0)
    density = vertical_edges / edges.size

    if density > 0.01:
        return "Chart likely uses candlesticks; candlestick patterns may be relevant."
    return "Chart may be line-based or mixed; candlestick patterns less explicit."


def mode_weights(mode):
    if mode == "Scalper":
        return 0.55, 0.45
    elif mode == "Swing":
        return 0.70, 0.30
    elif mode == "Position":
        return 0.85, 0.15
    return 0.70, 0.30


def mode_interpretation(mode, bullish, bearish):
    if mode == "Scalper":
        return "Scalper bias: Quick upside momentum likely." if bullish > bearish else "Scalper bias: Short-term downside pressure dominating."
    elif mode == "Swing":
        return "Swing bias: Market structure leans bullish." if bullish > bearish else "Swing bias: Bearish structure forming."
    elif mode == "Position":
        return "Position bias: Macro trend supports long exposure." if bullish > bearish else "Position bias: Macro trend favors defensive posture."
    return ""


# --- 5. CAMERA & UPLOAD ---
tab1, tab2 = st.tabs(["📷 Use Web/Phone Camera", "📁 Upload Image File"])

uploaded_file = None

with tab1:
    camera_image = st.camera_input("Position the trading chart clearly in front of the lens")
    if camera_image:
        uploaded_file = camera_image

with tab2:
    file_image = st.file_uploader("Choose a chart screenshot (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if file_image:
        uploaded_file = file_image

# --- 6. MAIN PROCESSING ---
if uploaded_file is not None:
    st.image(uploaded_file, caption="Captured Chart Target", use_container_width=True)

    overlay_img, trend_info, cv_img = detect_trendlines(uploaded_file)
    panel_description = describe_panels(uploaded_file)
    candle_comment = candlestick_hint(uploaded_file)

    trend_label = classify_trend(trend_info)
    risk_score = compute_risk_score(trend_info, cv_img)
    confidence_score = compute_confidence(trend_info, panel_description)
    sr_comment = detect_support_resistance(trend_info)

    st.subheader("📐 Detected Structural Features")
    st.image(overlay_img, caption="Trendline Detection Overlay", use_container_width=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Trend", trend_label)
    with col_b:
        st.metric("Risk", f"{risk_score}/100")
    with col_c:
        st.metric("Confidence", f"{confidence_score}/100")

    st.write(panel_description)
    st.write(sr_comment)
    st.write(candle_comment)

    if st.button("🚀 Calculate AI Sentiment Percentages"):
        with st.spinner("Analyzing chart patterns, price action, and indicator levels..."):
            try:
                mime = uploaded_file.type or "image/jpeg"
                file_bytes = uploaded_file.read()
                uploaded_file.seek(0)
                base64_image = base64.b64encode(file_bytes).decode("utf-8")

                trend_summary = (
                    f"Detected {len(trend_info)} major lines with slopes: "
                    f"{[round(t['slope'], 2) for t in trend_info]}"
                )

                analysis_prompt = (
                    f"You are analyzing this chart for a {mode.lower()} trader. "
                    "Provide a concise 3-sentence structural breakdown. "
                    f"Trend: {trend_label}. Risk: {risk_score}. Confidence: {confidence_score}. "
                    f"{panel_description} {sr_comment} {candle_comment}. {trend_summary}. "
                    "Then output ONLY this JSON block:\n"
                    "```json\n{\"bullish_pct\": X, \"bearish_pct\": Y}\n```"
                )

                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": analysis_prompt
                        }
                    ],
                    max_tokens=400,
                )

                raw_text = response.choices[0].message["content"]

                explanation = re.sub(r"```json.*?```", "", raw_text, flags=re.DOTALL).strip()
                st.subheader("📝 Market Analysis Summary")
                st.write(explanation)

                json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
                if json_match:
                    metrics = json.loads(json_match.group(1))
                    model_bullish = float(metrics.get("bullish_pct", 50))
                    model_bearish = float(metrics.get("bearish_pct", 50))
                    model_bullish, model_bearish = normalize_to_100(model_bullish, model_bearish)

                    rule_bullish, rule_bearish = rule_based_sentiment(trend_info)
                    rule_bullish, rule_bearish = normalize_to_100(rule_bullish, rule_bearish)

                    model_w, rule_w = mode_weights(mode)

                    base_bullish = model_bullish * model_w + rule_bullish * rule_w
                    base_bearish = 100 - base_bullish

                    conf_factor = confidence_score / 100.0
                    final_bullish = 50 + (base_bullish - 50) * conf_factor
                    final_bearish = 100 - final_bullish

                    final_bullish = round(final_bullish, 2)
                    final_bearish = round(final_bearish, 2)

                    mode_comment = mode_interpretation(mode, final_bullish, final_bearish)

                    st.subheader("📊 Sentiment Breakdown")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("**Model Sentiment**")
                        st.metric("🟢 Bullish", f"{model_bullish}%")
                        st.metric("🔴 Bearish", f"{model_bearish}%")

                    with col2:
                        st.markdown("**Rule-Based (Trendlines)**")
                        st.metric("🟢 Bullish", f"{rule_bullish}%")
                        st.metric("🔴 Bearish", f"{rule_bearish}%")

                    with col3:
                        st.markdown("**Ensemble (Mode + Confidence)**")
                        st.metric("🟢 Bullish", f"{final_bullish}%")
                        st.metric("🔴 Bearish", f"{final_bearish}%")
                        st.progress(final_bullish / 100.0)

                    st.subheader("🎯 Mode Interpretation")
                    st.write(mode_comment)

                else:
                    st.warning("Analysis complete, but JSON sentiment block could not be extracted.")
                    st.write(raw_text)

            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")

else:
    st.info("Use the camera or upload a chart image to begin analysis.")
