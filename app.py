import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Face Mask Detection AI",
    page_icon="FM",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* Hide Streamlit Elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Background — light base with flowing diagonal stripes */
.stApp {
    background-color: #F0F2F6;
    background-image:
        repeating-linear-gradient(
            135deg,
            transparent,
            transparent 40px,
            rgba(99,102,241,0.04) 40px,
            rgba(99,102,241,0.04) 80px
        ),
        repeating-linear-gradient(
            135deg,
            transparent,
            transparent 120px,
            rgba(16,185,129,0.035) 120px,
            rgba(16,185,129,0.035) 160px
        ),
        repeating-linear-gradient(
            135deg,
            transparent,
            transparent 200px,
            rgba(244,114,182,0.03) 200px,
            rgba(244,114,182,0.03) 240px
        );
    font-family: 'Inter', sans-serif;
}

/* Main Container */
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* Hero Section */
.hero-card {
    padding: 2.5rem 2.5rem 2rem;
    border-radius: 24px;
    background: linear-gradient(
        135deg,
        #ffffff 0%,
        #f8fafc 100%
    );
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow:
        0 4px 24px rgba(0,0,0,0.04),
        0 1px 4px rgba(0,0,0,0.03);
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

/* Decorative diagonal lines inside hero */
.hero-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        repeating-linear-gradient(
            135deg,
            transparent,
            transparent 30px,
            rgba(99,102,241,0.045) 30px,
            rgba(99,102,241,0.045) 31px
        );
    pointer-events: none;
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #1e293b;
    margin: 0 0 0.3rem;
    letter-spacing: -0.02em;
    position: relative;
}

.hero-subtitle {
    font-size: 1.15rem;
    color: #475569;
    font-weight: 500;
    margin: 0;
    position: relative;
}

.hero-badges {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-top: 1.2rem;
    flex-wrap: wrap;
    position: relative;
}

.hero-badge {
    padding: 7px 18px;
    border-radius: 999px;
    background: #EEF2FF;
    color: #4338CA;
    font-weight: 600;
    font-size: 0.85rem;
    border: 1px solid rgba(99,102,241,0.15);
}

/* Result Card */
.result-card {
    padding: 28px;
    border-radius: 20px;
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    text-align: center;
    margin-top: 10px;
}

.result-card h2 {
    color: #1e293b;
    font-weight: 700;
    letter-spacing: 0.02em;
}

.result-card h3 {
    color: #475569;
    font-weight: 500;
}

.result-card .result-icon {
    width: 56px;
    height: 56px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    font-weight: 800;
    margin-bottom: 8px;
}

.result-icon.mask {
    background: #D1FAE5;
    color: #065F46;
    border: 2px solid #6EE7B7;
}

.result-icon.no-mask {
    background: #FEE2E2;
    color: #991B1B;
    border: 2px solid #FCA5A5;
}

/* Feature Card */
.feature-card {
    padding: 18px 15px;
    border-radius: 16px;
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    text-align: center;
}

/* Upload Section */
.upload-box {
    padding: 20px;
    border-radius: 15px;
    border: 1px dashed rgba(0,0,0,0.15);
    margin-bottom: 20px;
}

/* Section headers */
.section-heading {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 0.5rem;
}

/* Override Streamlit markdown text colors for light mode */
.stMarkdown, .stMarkdown p, .stMarkdown li {
    color: #334155 !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #1e293b !important;
}

/* Override Streamlit subheader */
[data-testid="stSubheader"] {
    color: #1e293b !important;
}

/* Override metric labels */
[data-testid="stMetricLabel"] {
    color: #475569 !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #1e293b !important;
    font-weight: 700 !important;
}

/* Override info boxes */
[data-testid="stAlert"] {
    background: #ffffff !important;
    border: 1px solid rgba(0,0,0,0.06) !important;
    border-left: 4px solid #6366F1 !important;
    color: #334155 !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.03);
    border-radius: 12px !important;
}

[data-testid="stAlert"] p {
    color: #334155 !important;
}

/* Override Streamlit buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.25) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    box-shadow: 0 4px 16px rgba(99,102,241,0.35) !important;
    transform: translateY(-1px);
}

/* Override file uploader */
[data-testid="stFileUploader"] {
    background: #ffffff;
    border-radius: 16px;
    border: 2px dashed #CBD5E1;
    padding: 1rem;
}

[data-testid="stFileUploader"] label {
    color: #475569 !important;
}

/* Override dividers */
hr {
    border-color: rgba(0,0,0,0.06) !important;
}

/* Override progress bars */
[data-testid="stProgress"] > div > div {
    background-color: #E2E8F0 !important;
}

[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #6366F1, #818CF8) !important;
}

/* Override write/text */
[data-testid="stText"], .stWrite {
    color: #334155 !important;
}

/* Footer */
.footer-section {
    text-align: center;
    color: #94A3B8;
    padding: 20px;
    font-size: 0.9rem;
    font-weight: 500;
}

.footer-section strong {
    color: #64748B;
}

/* Override Streamlit spinner */
[data-testid="stSpinner"] {
    color: #475569 !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# MODEL LOADING
# =====================================================

@st.cache_resource
def load_model():
    model = torch.load(
        "face_mask_model_full.pth",
        map_location="cpu",
        weights_only=False
    )

    model.eval()

    return model

model = load_model()

# =====================================================
# TRANSFORM
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

classes = [
    "with_mask",
    "without_mask"
]

# =====================================================
# HERO SECTION
# =====================================================

st.markdown("""
<div class="hero-card">

<h1 class="hero-title">
Face Mask Detection AI
</h1>

<p class="hero-subtitle">
Deep Learning Powered by ResNet18
</p>

<div class="hero-badges">

<span class="hero-badge">
PyTorch
</span>

<span class="hero-badge">
98.74% Validation Accuracy
</span>

<span class="hero-badge">
7,553 Images
</span>

</div>

</div>
""", unsafe_allow_html=True)

# =====================================================
# FEATURE CARDS
# =====================================================

f1, f2, f3 = st.columns(3)

with f1:
    st.info("Fast Inference")

with f2:
    st.info("Transfer Learning")

with f3:
    st.info("98.74% Accuracy")

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# UPLOAD SECTION
# =====================================================

st.markdown("""
### Upload Image

Upload a face image and let the AI determine whether
a face mask is present.

**Supported Formats**
- JPG
- JPEG
- PNG
""")

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# PREDICTION
# =====================================================

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    analyze = st.button(
        "Analyze Image",
        use_container_width=True
    )

    if analyze:

        col1, col2 = st.columns([1.2, 1])

        with col1:

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            img_tensor = transform(
                image
            ).unsqueeze(0)

            with st.spinner(
                "Analyzing image..."
            ):

                with torch.no_grad():

                    outputs = model(
                        img_tensor
                    )

                    probabilities = torch.softmax(
                        outputs,
                        dim=1
                    )

                    confidence, prediction = torch.max(
                        probabilities,
                        1
                    )

                    mask_prob = (
                        probabilities[0][0].item()
                        * 100
                    )

                    no_mask_prob = (
                        probabilities[0][1].item()
                        * 100
                    )

            predicted_class = classes[
                prediction.item()
            ]

            confidence_score = (
                confidence.item()
                * 100
            )

            if predicted_class == "with_mask":

                st.markdown(f"""
                <div class="result-card">

                <div class="result-icon mask">OK</div>

                <h2>
                MASK DETECTED
                </h2>

                <h3>
                {confidence_score:.2f}% Confidence
                </h3>

                </div>
                """,
                unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="result-card">

                <div class="result-icon no-mask">!!</div>

                <h2>
                NO MASK DETECTED
                </h2>

                <h3>
                {confidence_score:.2f}% Confidence
                </h3>

                </div>
                """,
                unsafe_allow_html=True)

            st.markdown("### Confidence Score")

            st.progress(
                int(confidence_score)
            )

            st.markdown(
                f"""
                <h2 style='text-align:center;color:#1e293b;'>
                {confidence_score:.2f}%
                </h2>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                "### Prediction Breakdown"
            )

            st.write(
                f"Mask: {mask_prob:.2f}%"
            )

            st.progress(
                int(mask_prob)
            )

            st.write(
                f"No Mask: {no_mask_prob:.2f}%"
            )

            st.progress(
                int(no_mask_prob)
            )

# =====================================================
# MODEL METRICS
# =====================================================

st.divider()

st.subheader("Model Performance")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "Validation Accuracy",
        "98.74%"
    )

with m2:
    st.metric(
        "Training Images",
        "6,042"
    )

with m3:
    st.metric(
        "Model",
        "ResNet18"
    )

# =====================================================
# LIMITATIONS
# =====================================================

st.divider()

st.subheader("Known Limitations")

st.info("""
- Best performance on face-centric images.

- Full-body images may reduce accuracy because the model classifies the entire image rather than detecting and cropping faces.

- Rotated images (90 or upside-down) may produce incorrect predictions.

- Multiple faces in a single image are not explicitly supported.

- Future versions will integrate OpenCV face detection and real-time webcam inference.
""")

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown("""
<div class="footer-section">

<a href="https://github.com/pranjalbhadani/FaceMaskDetection" target="_blank" rel="noopener noreferrer" style="
    display:inline-flex;
    align-items:center;
    gap:8px;
    padding:8px 20px;
    border-radius:999px;
    background:#1e293b;
    color:#ffffff;
    text-decoration:none;
    font-weight:600;
    font-size:0.9rem;
    transition:all 0.2s ease;
    box-shadow:0 2px 8px rgba(0,0,0,0.1);
    margin-bottom:16px;
" onmouseover="this.style.background='#334155';this.style.boxShadow='0 4px 16px rgba(0,0,0,0.15)';this.style.transform='translateY(-1px)'" onmouseout="this.style.background='#1e293b';this.style.boxShadow='0 2px 8px rgba(0,0,0,0.1)';this.style.transform='translateY(0)'">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
    View on GitHub
</a>

<br>

Built with <strong>PyTorch</strong> / <strong>ResNet18</strong> / <strong>Streamlit</strong>

<br>

Developed by <strong>Pranjal Bhadani</strong>

</div>
""", unsafe_allow_html=True)