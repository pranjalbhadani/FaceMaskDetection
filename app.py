import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Face Mask Detection AI",
    page_icon="😷",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* Hide Streamlit Elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Background */
.stApp {
    background-color: #0E1117;
}

/* Main Container */
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* Hero Section */
.hero-card {
    padding: 2.5rem;
    border-radius: 24px;
    background: linear-gradient(
        135deg,
        #111827 0%,
        #1f2937 100%
    );
    border: 1px solid rgba(255,255,255,0.08);
    text-align: center;
    margin-bottom: 2rem;
}

/* Result Card */
.result-card {
    padding: 25px;
    border-radius: 20px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    text-align: center;
    backdrop-filter: blur(10px);
    margin-top: 10px;
}

/* Feature Card */
.feature-card {
    padding: 15px;
    border-radius: 15px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    text-align: center;
}

/* Upload Section */
.upload-box {
    padding: 20px;
    border-radius: 15px;
    border: 1px dashed rgba(255,255,255,0.15);
    margin-bottom: 20px;
}

/* Footer */
.footer {
    text-align:center;
    color:gray;
    padding:20px;
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

<h1 style="font-size:3rem;">
😷 Face Mask Detection AI
</h1>

<p style="font-size:1.2rem;color:#C0C0C0;">
Deep Learning Powered by ResNet18
</p>

<div style="
display:flex;
justify-content:center;
gap:15px;
margin-top:20px;
flex-wrap:wrap;
">

<span style="
padding:8px 16px;
border-radius:999px;
background:#374151;
">
PyTorch
</span>

<span style="
padding:8px 16px;
border-radius:999px;
background:#374151;
">
98.74% Validation Accuracy
</span>

<span style="
padding:8px 16px;
border-radius:999px;
background:#374151;
">
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
    st.info("⚡ Fast Inference")

with f2:
    st.info("🧠 Transfer Learning")

with f3:
    st.info("🎯 98.74% Accuracy")

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# UPLOAD SECTION
# =====================================================

st.markdown("""
### 📤 Upload Image

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
        "🚀 Analyze Image",
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

                <h1>😷</h1>

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

                <h1>⚠️</h1>

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
                <h2 style='text-align:center'>
                {confidence_score:.2f}%
                </h2>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                "### Prediction Breakdown"
            )

            st.write(
                f"😷 Mask: {mask_prob:.2f}%"
            )

            st.progress(
                int(mask_prob)
            )

            st.write(
                f"⚠️ No Mask: {no_mask_prob:.2f}%"
            )

            st.progress(
                int(no_mask_prob)
            )

# =====================================================
# MODEL METRICS
# =====================================================

st.divider()

st.subheader("📊 Model Performance")

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

st.subheader("⚠️ Known Limitations")

st.info("""
• Best performance on face-centric images.

• Full-body images may reduce accuracy because the model classifies the entire image rather than detecting and cropping faces.

• Rotated images (90° or upside-down) may produce incorrect predictions.

• Multiple faces in a single image are not explicitly supported.

• Future versions will integrate OpenCV face detection and real-time webcam inference.
""")

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown("""
<div class="footer">

Built with PyTorch • ResNet18 • Streamlit

<br>

Developed by Pranjal Bhadani

</div>
""", unsafe_allow_html=True)