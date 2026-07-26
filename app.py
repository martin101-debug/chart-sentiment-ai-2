import os
import base64
import json
import re
import io

import streamlit as st
import openai
import numpy as np
from PIL import Image

# --- 1. APP HEADER ---
st.set_page_config(page_title="AI Chart Sentiment Scanner", layout="centered")
st.title("📊 Professional Chart AI Sentiment Scanner")
st.write("Upload or capture a trading chart to estimate Bullish vs. Bearish sentiment using GPT‑4o.")

# --- 2. API KEY HANDLING ---
st.sidebar.header("🔑 Authentication")

env_api_key = os.getenv("OPENAI_API_KEY")
api_key_input = st.sidebar.text_input(
    "Enter OpenAI API Key (optional if set in Secrets)",
    type="password",
    value="" if env_api_key else "",
)

api_key = api_key_input or env_api_key

if not api_key:
    st.info("Please enter your OpenAI API Key or configure OPENAI_API_KEY in Streamlit Secrets.")
    st.stop()

openai.api_key = api_key

# --- 3. TRADING MODE ---
st.subheader("🎛️ Trading Style Mode")
mode = st.selectbox("Choose analysis mode", ["Scalper", "Swing", "Position"], index=1)

# --- 4. SIMPLE IMAGE FEATURES (NO OPENCV) ---
def simple_edge_density(img: Image.Image):
    """Simple edge density using NumPy gradients."""
    gray = np.mean(np.array(img), axis=2)
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx**2 + gy**2)
    return float(np.mean(mag > np.percentile(mag, 90)))

def classify_trend(edge_density):
    if edge_density > 0.12:
        return "Strong Trend"
    elif edge_density > 0.06:
        return "Mild Trend"
    else:
        return "Sideways / Range"

def compute_risk(edge_density):
    return int(min(100, max(0, edge_density * 300)))

def compute_confidence(edge_density):
    return int(min(100, 40 + edge_density * 200))

# --- 5. CAMERA & UPLOAD ---
tab1, tab2 = st.tabs(["📷 Use Camera", "📁 Upload Image"])

uploaded_file = None

with tab1:
    cam = st.camera_input("Capture chart")
    if cam:
        uploaded_file = cam

with tab2:
    up = st.file_uploader("Upload chart image", type=["png", "jpg", "jpeg"])
    if up:
        uploaded_file = up

# --- 6. MAIN PROCESSING ---
if uploaded_file is not None:
    st.image(uploaded_file, caption="Chart", use_container_width=True)

    img = Image.open(uploaded_file).convert("RGB")

    edge_density = simple_edge_density(img)
    trend_label = classify_trend(edge_density)
    risk_score = compute_risk(edge_density)
    confidence_score = compute_confidence(edge_density)

    st.subheader("📐 Structural Features")
    col1, col2, col3 = st.columns(3)
    col1.metric("Trend", trend_label)
    col2.metric("Risk", f"{risk_score}/100")
    col3.metric("Confidence", f"{confidence_score}/100")

    if st.button("🚀 Calculate AI Sentiment"):
        with st.spinner("Analyzing chart with GPT‑4o…"):
            try:
                mime = uploaded_file.type or "image/jpeg"
                file_bytes = uploaded_file.read()
                uploaded_file.seek(0)
                base64_image = base64.b64encode(file_bytes).decode("utf-8")

                prompt = (
                    f"You are analyzing this chart for a {mode.lower()} trader. "
                    f"Trend: {trend_label}. Risk: {risk_score}. Confidence: {confidence_score}. "
                    "Provide a concise 3-sentence structural breakdown. "
                    "Then output ONLY this JSON block:\n"
                    "```json\n{\"bullish_pct\": X, \"bearish_pct\": Y}\n```"
                )

                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime};base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=400,
                )

                raw = response.choices[0].message["content"]

                explanation = re.sub(r"```json.*?```", "", raw, flags=re.DOTALL).strip()
                st.subheader("📝 Market Summary")
                st.write(explanation)

                json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
                if json_match:
                    metrics = json.loads(json_match.group(1))
                    bull = float(metrics.get("bullish_pct", 50))
                    bear = float(metrics.get("bearish_pct", 50))

                    bull = round(bull, 2)
                    bear = round(bear, 2)

                    st.subheader("📊 Sentiment")
                    colA, colB = st.columns(2)
                    colA.metric("🟢 Bullish", f"{bull}%")
                    colB.metric("🔴 Bearish", f"{bear}%")
                    st.progress(bull / 100)

                else:
                    st.warning("Could not extract JSON block.")
                    st.write(raw)

            except Exception as e:
                st.error(f"Error: {str(e)}")

else:
    st.info("Upload or capture a chart to begin.")
