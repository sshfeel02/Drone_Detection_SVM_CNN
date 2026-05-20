import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog
import io

# ============================================================
# 페이지 설정
# ============================================================

st.set_page_config(
    page_title="공중 물체 분류기",
    page_icon="✈️",
    layout="centered"
)

# ============================================================
# 스타일
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Noto+Sans+KR:wght@300;400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', sans-serif;
    }
    .main {
        background-color: #0a0e1a;
    }
    .stApp {
        background-color: #0a0e1a;
        color: #e0e6f0;
    }
    h1, h2, h3 {
        font-family: 'Space Mono', monospace !important;
        color: #00d4ff !important;
    }
    .title-box {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-bottom: 1px solid #1e2d4a;
        margin-bottom: 2rem;
    }
    .title-box h1 {
        font-size: 2rem;
        letter-spacing: 0.1em;
        margin: 0;
    }
    .title-box p {
        color: #6a8aaa;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    .result-box {
        background: #0f1b2d;
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    .model-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #6a8aaa;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.3rem;
    }
    .prediction {
        font-family: 'Space Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: #00d4ff;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .confidence {
        font-size: 0.85rem;
        color: #4a7a9b;
        margin-top: 0.2rem;
    }
    .divider {
        border: none;
        border-top: 1px solid #1e2d4a;
        margin: 1rem 0;
    }
    .upload-hint {
        color: #4a7a9b;
        font-size: 0.85rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    .stProgress > div > div {
        background-color: #00d4ff !important;
    }
    .class-tag {
        display: inline-block;
        background: #0f2a40;
        border: 1px solid #1e4a6a;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.8rem;
        color: #6ab4d4;
        margin: 0.2rem;
        font-family: 'Space Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 설정
# ============================================================

IMG_SIZE = (48, 48)

CLASS_EMOJI = {
    "airplane":   "✈️",
    "helicopter": "🚁",
    "bird":       "🐦",
    "drone":      "🛸"
}

# ============================================================
# 모델 로드
# ============================================================

@st.cache_resource
def load_models():
    cnn_model   = tf.keras.models.load_model("cnn_model.keras")
    svm_model   = joblib.load("svm_model.pkl")
    class_names = joblib.load("class_names.pkl")
    return cnn_model, svm_model, class_names

# ============================================================
# HOG 특징 추출
# ============================================================

def extract_hog_feature(img_array):
    if img_array.ndim == 3:
        gray = rgb2gray(img_array)
    else:
        gray = img_array
    gray = resize(gray, IMG_SIZE, anti_aliasing=True)
    feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True
    )
    return feat

# ============================================================
# 예측 함수
# ============================================================

def predict(img_pil, cnn_model, svm_model, class_names):
    # CNN 예측
    img_resized = img_pil.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img_resized, dtype=np.float32)
    arr_batch = np.expand_dims(arr, axis=0)
    cnn_prob = cnn_model.predict(arr_batch, verbose=0)[0]
    cnn_idx  = int(np.argmax(cnn_prob))

    # SVM 예측
    img_np  = np.array(img_pil.convert("RGB").resize(IMG_SIZE))
    hog_feat = extract_hog_feature(img_np).reshape(1, -1)
    svm_idx  = int(svm_model.predict(hog_feat)[0])
    svm_prob = svm_model.predict_proba(hog_feat)[0]

    return {
        "cnn_class": class_names[cnn_idx],
        "cnn_prob":  float(cnn_prob[cnn_idx]),
        "cnn_all":   {class_names[i]: float(cnn_prob[i]) for i in range(len(class_names))},
        "svm_class": class_names[svm_idx],
        "svm_prob":  float(svm_prob[svm_idx]),
        "svm_all":   {class_names[i]: float(svm_prob[i]) for i in range(len(class_names))},
    }

# ============================================================
# UI
# ============================================================

st.markdown("""
<div class="title-box">
    <h1>✈️ AIRBORNE CLASSIFIER</h1>
    <p>CNN + SVM 기반 공중 물체 분류기</p>
</div>
""", unsafe_allow_html=True)

# 지원 클래스 표시
st.markdown("""
<div style="text-align:center; margin-bottom: 1.5rem;">
    <span class="class-tag">✈️ airplane</span>
    <span class="class-tag">🚁 helicopter</span>
    <span class="class-tag">🐦 bird</span>
    <span class="class-tag">🛸 drone</span>
</div>
""", unsafe_allow_html=True)

# 모델 로드
try:
    cnn_model, svm_model, class_names = load_models()
    st.success("✅ 모델 로드 완료")
except Exception as e:
    st.error(f"❌ 모델 로드 실패: {e}")
    st.info("cnn_model.keras / svm_model.pkl / class_names.pkl 파일이 같은 폴더에 있는지 확인하세요.")
    st.stop()

# 파일 업로드
uploaded_file = st.file_uploader(
    "이미지를 업로드하세요",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="분류할 공중 물체 이미지를 선택하세요"
)
st.markdown('<p class="upload-hint">지원 형식: JPG, PNG, BMP, WEBP</p>', unsafe_allow_html=True)

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file)

    # 이미지 표시
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img_pil, caption="업로드된 이미지", use_container_width=True)

    # 예측
    with st.spinner("분류 중..."):
        result = predict(img_pil, cnn_model, svm_model, class_names)

    # 결과 출력
    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    col_cnn, col_svm = st.columns(2)

    with col_cnn:
        emoji = CLASS_EMOJI.get(result["cnn_class"], "🔍")
        st.markdown(f'<p class="model-label">CNN 예측</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="prediction">{emoji} {result["cnn_class"]}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="confidence">신뢰도: {result["cnn_prob"]*100:.1f}%</p>', unsafe_allow_html=True)

    with col_svm:
        emoji = CLASS_EMOJI.get(result["svm_class"], "🔍")
        st.markdown(f'<p class="model-label">SVM 예측</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="prediction">{emoji} {result["svm_class"]}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="confidence">신뢰도: {result["svm_prob"]*100:.1f}%</p>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # CNN 전체 확률
    st.markdown('<p class="model-label">CNN 클래스별 확률</p>', unsafe_allow_html=True)
    for cls, prob in sorted(result["cnn_all"].items(), key=lambda x: x[1], reverse=True):
        emoji = CLASS_EMOJI.get(cls, "")
        st.markdown(f"**{emoji} {cls}**")
        st.progress(prob)
        st.markdown(f"<p style='color:#4a7a9b; font-size:0.8rem; margin-top:-0.5rem;'>{prob*100:.1f}%</p>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
