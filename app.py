"""
PlantDoc AI web app.
Run: streamlit run app.py
"""

import json

import streamlit as st
from PIL import Image, UnidentifiedImageError

import config
from src.utils import pretty_class_name

st.set_page_config(page_title="PlantDoc AI", page_icon="🌿", layout="centered")

LOW_CONFIDENCE = 0.60  # below this, the app tells the user the result is uncertain


@st.cache_resource(show_spinner="Loading model...")
def get_predictor():
    """Runs once per app session; later uploads reuse the loaded model."""
    from predict import Predictor  # imported here so the page can show errors before torch loads

    return Predictor()


def load_test_summary() -> dict | None:
    """Real test results from evaluate.py, if it has been run."""
    path = config.OUTPUTS_DIR / "test_summary.json"
    return json.loads(path.read_text()) if path.exists() else None


# ---------- Header ----------
st.title("🌿 PlantDoc AI")
st.write("Upload a photo of a single plant leaf. The model predicts the disease, or tells you if the leaf looks healthy.")

# ---------- Model loading with clear errors ----------
if not config.MODEL_PATH.exists():
    st.error(
        f"No trained model found at `{config.MODEL_PATH.relative_to(config.PROJECT_ROOT)}`.\n\n"
        "Train one first by running `python train.py` in the terminal, then refresh this page."
    )
    st.stop()

try:
    predictor = get_predictor()
except Exception as error:
    st.error(f"The model file exists but couldn't be loaded: {error}")
    st.stop()

# ---------- Sidebar: model details ----------
with st.sidebar:
    st.header("About the model")
    st.write(f"Classes: **{len(predictor.class_names)}**")
    st.write("Architecture: CNN trained from scratch")
    summary = load_test_summary()
    if summary:
        st.write(f"Test accuracy: **{summary['test_accuracy']:.1%}**")
        st.caption(f"Measured on {summary['test_images']:,} held-out PlantVillage images.")
    else:
        st.caption("Run `python evaluate.py` to show test accuracy here.")

    with st.expander("Plants and diseases it knows"):
        for name in predictor.class_names:
            st.write(f"- {pretty_class_name(name)}")

# ---------- Upload ----------
uploaded = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Tip: photograph one leaf, filling most of the frame, on a plain background.")
else:
    try:
        image = Image.open(uploaded)
        image.load()  # catches corrupted or cut-off files
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("This file couldn't be read as an image. Please upload a valid JPG or PNG.")
        st.stop()

    image_col, result_col = st.columns(2)  # side by side on desktop, stacked on phones

    with image_col:
        st.image(image, caption=uploaded.name)

    with result_col:
        with st.spinner("Analyzing leaf..."):
            results = predictor.predict(image, top_k=3)

        top_name, top_prob = results[0]
        st.subheader(pretty_class_name(top_name))

        if "healthy" in pretty_class_name(top_name).lower():
            st.success("The leaf looks healthy.")
        else:
            st.warning("Possible disease detected.")

        st.progress(top_prob, text=f"Confidence: {top_prob:.1%}")

        if top_prob < LOW_CONFIDENCE:
            st.info(
                "The model isn't sure about this one. Try a sharper photo of a single leaf, "
                "or this plant may not be one it was trained on."
            )

    st.markdown("#### Top 3 predictions")
    for name, prob in results:
        st.progress(prob, text=f"{pretty_class_name(name)}: {prob:.1%}")

    with st.expander("How to read the confidence score"):
        st.write(
            "The confidence is the model's own probability for each class, and the three values "
            "come from the same set that adds up to 100% across all classes. A high score means the "
            "image closely matches what the model saw in training. It doesn't guarantee the "
            "diagnosis is right: the model always picks one of its known classes, even for a plant "
            "or problem it has never seen. Two close top scores mean it's torn between those classes."
        )

st.divider()
st.caption(
    "Disclaimer: predictions are model estimates based on lab photos from the PlantVillage dataset. "
    "They should not replace a diagnosis from an agricultural expert or plant pathologist."
)