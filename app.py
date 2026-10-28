"""
PlantDoc AI web app.
Run: streamlit run app.py
"""

import html
import json
from collections import defaultdict

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

import config
from src.disease_info import get_info, related_conditions
from src.insights import (
    build_report,
    caveats,
    confidence_band,
    load_class_reliability,
    overlay_heatmap,
    photo_quality,
)
from src.utils import pretty_class_name

st.set_page_config(page_title="PlantDoc AI", page_icon="🌿", layout="wide")

CATEGORY_ICONS = {"Healthy": "✅", "Fungal": "🍄", "Bacterial": "🦠", "Viral": "🧬", "Oomycete": "💧", "Pest": "🕷️"}
IMPACT_COLORS = {"None": "green", "Low": "blue", "Moderate": "orange", "High": "red"}
BAND_COLORS = {"High": "green", "Medium": "orange", "Low": "red"}

st.html("""
<style>
  .block-container { padding-top: 3.5rem; max-width: 1200px; }
  .hero {
    background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 55%, #558b2f 100%);
    color: #fff; border-radius: 18px; padding: 1.8rem 2rem; margin-bottom: 1.2rem;
  }
  .hero h1 { color: #fff; margin: 0; padding: 0; font-size: 2.1rem; }
  .hero p { margin: .35rem 0 1rem; opacity: .92; font-size: 1.05rem; }
  .chip {
    display: inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.25);
    border-radius: 999px; padding: .2rem .75rem; margin: .15rem .3rem .15rem 0; font-size: .85rem;
  }
  .result-title { font-size: 1.9rem; font-weight: 700; line-height: 1.2; margin: .1rem 0 .2rem; }
  .result-plant { text-transform: uppercase; letter-spacing: .08em; font-size: .8rem; opacity: .7; }
  .pathogen { font-style: italic; opacity: .75; margin-bottom: .6rem; }
  .bar-row { margin: .45rem 0; }
  .bar-label { display: flex; justify-content: space-between; font-size: .92rem; margin-bottom: .2rem; }
  .bar-track { background: rgba(128,128,128,.18); border-radius: 999px; height: 10px; overflow: hidden; }
  .bar-fill { background: linear-gradient(90deg, #66bb6a, #2e7d32); height: 100%; border-radius: 999px; }
  .bar-fill.other { background: rgba(128,128,128,.55); }
  .step-num {
    display: inline-flex; align-items: center; justify-content: center; width: 1.8rem; height: 1.8rem;
    border-radius: 50%; background: #2e7d32; color: #fff; font-weight: 700; margin-bottom: .4rem;
  }
</style>
""")


# ---------- Cached resources ----------

@st.cache_resource(show_spinner="Loading model...")
def get_predictor():
    """Runs once per app session; later uploads reuse the loaded model."""
    from predict import Predictor  # imported here so the page can show errors before torch loads

    return Predictor()


@st.cache_data
def load_test_summary() -> dict | None:
    """Real test results from evaluate.py, if it has been run."""
    path = config.OUTPUTS_DIR / "test_summary.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_reliability(class_names: tuple[str, ...]):
    return load_class_reliability(list(class_names))


@st.cache_data
def load_test_samples() -> pd.DataFrame:
    """Test-set images the model never trained on, for the 'Try a sample' tab."""
    splits_csv = config.SPLITS_DIR / "splits.csv"
    if not splits_csv.exists() or not config.DATASET_DIR.exists():
        return pd.DataFrame(columns=["path", "label"])
    df = pd.read_csv(splits_csv, dtype=str, keep_default_na=False, usecols=["path", "label", "split"])
    return df[df["split"] == "test"][["path", "label"]].reset_index(drop=True)


def bar(label: str, prob: float, highlight: bool) -> str:
    css = "bar-fill" if highlight else "bar-fill other"
    return (
        f'<div class="bar-row"><div class="bar-label"><span>{html.escape(label)}</span><b>{prob:.1%}</b></div>'
        f'<div class="bar-track"><div class="{css}" style="width:{max(prob, 0.005):.1%}"></div></div></div>'
    )


def bullet_list(items: list[str]) -> None:
    st.markdown("\n".join(f"- {item}" for item in items))


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

class_names = predictor.class_names
summary = load_test_summary()
reliability = load_reliability(tuple(class_names))
plants = defaultdict(list)
for name in class_names:
    info = get_info(name)
    plants[info.plant].append(info.condition)

# ---------- Header ----------
chips = [f"{len(plants)} plants", f"{len(class_names)} classes", "CNN trained from scratch"]
if summary:
    chips.insert(0, f"{summary['test_accuracy']:.1%} test accuracy")
st.html(
    '<div class="hero"><h1>🌿 PlantDoc AI</h1>'
    "<p>Upload a photo of a plant leaf to identify diseases and get symptoms, causes, treatment and prevention advice.</p>"
    + "".join(f'<span class="chip">{html.escape(c)}</span>' for c in chips)
    + "</div>"
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Predictions to show", 3, 10, 5)
    show_heatmap = st.toggle("Show where the model looked", value=True,
                             help="Grad-CAM heatmap: red areas influenced the prediction most.")

    st.header("📊 About the model")
    if summary:
        col_a, col_b = st.columns(2)
        col_a.metric("Accuracy", f"{summary['test_accuracy']:.1%}")
        col_b.metric("Macro F1", f"{summary['macro_f1']:.3f}")
        st.caption(
            f"Measured on {summary['test_images']:,} held-out PlantVillage images. "
            f"{summary['wrong_predictions']} were misclassified."
        )
    else:
        st.caption("Run `python evaluate.py` to show test accuracy here.")

    with st.expander(f"🌱 Plants and conditions it knows ({len(class_names)})"):
        for plant, conditions in sorted(plants.items()):
            st.markdown(f"**{plant}**: {', '.join(conditions)}")

# ---------- Input ----------
upload_tab, camera_tab, sample_tab = st.tabs(["📤 Upload", "📷 Camera", "🎲 Try a sample"])
source, source_name, true_label = None, None, None

with upload_tab:
    uploaded = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded is not None:
        source, source_name = uploaded, uploaded.name

with camera_tab:
    photo = st.camera_input("Take a photo of a leaf", label_visibility="collapsed")
    if photo is not None and source is None:
        source, source_name = photo, "camera photo"

with sample_tab:
    samples = load_test_samples()
    if samples.empty:
        st.info("Sample images need the dataset in `data/plantvillage/` and a saved split. Run `python train.py` first.")
    else:
        st.caption("Pick an image from the test set. The model never saw these during training.")
        pick_col, button_col = st.columns([3, 1], vertical_alignment="bottom")
        options = ["Any class"] + sorted(samples["label"].unique(), key=pretty_class_name)
        chosen = pick_col.selectbox("Class", options, format_func=lambda n: n if n == "Any class" else pretty_class_name(n))
        if button_col.button("🎲 Random image", width="stretch"):
            pool = samples if chosen == "Any class" else samples[samples["label"] == chosen]
            st.session_state["sample"] = pool.sample(1).iloc[0].to_dict()
        if "sample" in st.session_state and source is None:
            sample = st.session_state["sample"]
            source = config.DATASET_DIR / sample["path"]
            source_name, true_label = source.name, sample["label"]

# ---------- Empty state ----------
if source is None:
    st.write("")
    steps = [
        ("Photograph one leaf", "Fill most of the frame with a single leaf, on a plain background if you can."),
        ("Upload or snap", "Use a JPG or PNG, take a photo with your camera, or try a test-set sample."),
        ("Read the diagnosis", "See the condition, confidence, symptoms, treatment and where the model looked."),
    ]
    for col, (number, (title, text)) in zip(st.columns(3), enumerate(steps, 1)):
        with col.container(border=True):
            st.html(f'<div class="step-num">{number}</div>')
            st.markdown(f"**{title}**")
            st.caption(text)
    st.stop()

# ---------- Prediction ----------
try:
    image = Image.open(source)
    image.load()  # catches corrupted or cut-off files
    image = image.convert("RGB")
except (UnidentifiedImageError, OSError):
    st.error("This file couldn't be read as an image. Please upload a valid JPG or PNG.")
    st.stop()

with st.spinner("Analyzing leaf..."):
    results = predictor.predict(image, top_k=top_k)
    top_name, top_prob = results[0]
    second_name, second_prob = results[1]
    cam = predictor.heatmap(image, top_name) if show_heatmap else None

info = get_info(top_name)
band, band_text = confidence_band(top_prob)
checks = photo_quality(image)
notes = caveats(top_name, top_prob, second_prob, get_info(second_name).plant == info.plant)
rel = reliability.get(top_name)

st.divider()
image_col, result_col = st.columns([5, 7], gap="large")

with image_col:
    if cam is not None:
        original_tab, heat_tab = st.tabs(["Photo", "Where the model looked"])
        original_tab.image(image, caption=source_name, width="stretch")
        with heat_tab:
            st.image(overlay_heatmap(image, cam), width="stretch")
            st.caption("Red and yellow areas influenced the prediction most; blue areas the least.")
    else:
        st.image(image, caption=source_name, width="stretch")

    if true_label:
        correct = true_label == top_name
        (st.success if correct else st.error)(
            f"True class: **{pretty_class_name(true_label)}** {'✓ correct' if correct else '✗ wrong'}"
        )

with result_col:
    st.html(
        f'<div class="result-plant">{html.escape(info.plant)}</div>'
        f'<div class="result-title">{html.escape(info.condition)}</div>'
        + (f'<div class="pathogen">{html.escape(info.pathogen)}</div>' if info.pathogen else "")
    )
    category_color = "green" if info.is_healthy else "violet"
    st.markdown(
        f":{category_color}-badge[{CATEGORY_ICONS.get(info.category, '❔')} {info.category}] "
        f":{IMPACT_COLORS.get(info.impact, 'gray')}-badge[Impact: {info.impact}] "
        f":{BAND_COLORS[band]}-badge[{band} confidence]"
    )

    if info.is_healthy:
        st.success("The leaf looks healthy.")
    else:
        st.warning("Possible disease or pest detected.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence", f"{top_prob:.1%}", help=band_text)
    m2.metric("Runner-up", f"{second_prob:.1%}", help=pretty_class_name(second_name))
    if rel:
        m3.metric("Class precision", f"{rel.precision:.1%}",
                  help="On the test set, how often a prediction of this class was correct.")
    st.caption(band_text)

    if info.summary:
        st.write(info.summary)
    for note in notes:
        st.info(note, icon="ℹ️")

# ---------- Details ----------
tab_names = ["🔍 Symptoms & causes", "💊 Treatment & prevention", "📈 Model insight", "📷 Photo check"]
if info.is_healthy:
    tab_names[:2] = ["🌱 Care tips", "👀 What to watch for"]
symptoms_tab, treatment_tab, model_tab, photo_tab = st.tabs(tab_names)

with symptoms_tab:
    if info.is_healthy:
        bullet_list(info.prevention)
    else:
        left, right = st.columns(2, gap="large")
        with left, st.container(border=True):
            st.markdown("##### What to look for")
            bullet_list(info.symptoms or ["No symptom description available."])
        with right, st.container(border=True):
            st.markdown("##### What makes it spread")
            bullet_list(info.conditions or ["No information available."])

with treatment_tab:
    if info.is_healthy:
        others = [get_info(n) for n in related_conditions(top_name, class_names)]
        if others:
            st.write(f"Other {info.plant.lower()} conditions the model can recognise. Keep an eye out for their early signs:")
            for other in others:
                with st.expander(f"{CATEGORY_ICONS.get(other.category, '')} {other.condition}"):
                    st.write(other.summary)
                    bullet_list(other.symptoms[:2])
        else:
            st.write("The model only knows healthy leaves for this plant, so it can't warn you about its diseases.")
    else:
        left, right = st.columns(2, gap="large")
        with left, st.container(border=True):
            st.markdown("##### Treatment")
            bullet_list(info.treatment or ["No treatment information available."])
        with right, st.container(border=True):
            st.markdown("##### Prevention")
            bullet_list(info.prevention or ["No prevention information available."])
        st.caption(
            "Product names are examples of active ingredients. Always follow the label and local regulations, "
            "and ask a local agricultural extension service for advice specific to your area."
        )

with model_tab:
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(f"##### Top {len(results)} predictions")
        st.html("".join(bar(pretty_class_name(n), p, i == 0) for i, (n, p) in enumerate(results)))
        st.caption("Probabilities across all classes add up to 100%.")
    with right:
        st.markdown("##### How reliable is this class?")
        if rel:
            c1, c2, c3 = st.columns(3)
            c1.metric("Precision", f"{rel.precision:.1%}", help="When the model predicted this class, how often it was right.")
            c2.metric("Recall", f"{rel.recall:.1%}", help="Of the test images of this class, how many it found.")
            c3.metric("Test images", f"{rel.test_images:,}")
            if rel.mistaken_for:
                st.write("On the test set, these classes were sometimes wrongly predicted as this one:")
                bullet_list([f"{pretty_class_name(n)}: {count} image{'s' * (count > 1)}" for n, count in rel.mistaken_for])
            else:
                st.write("No other class was wrongly predicted as this one on the test set.")
        else:
            st.caption("Run `python evaluate.py` to see per-class test results here.")

    with st.expander("How to read the confidence score"):
        st.write(
            "The confidence is the model's own probability for each class. A high score means the image closely "
            "matches what the model saw in training. It doesn't guarantee the diagnosis is right: the model always "
            "picks one of its known classes, even for a plant or problem it has never seen. Two close top scores "
            "mean it's torn between those classes."
        )
        st.markdown("- **Above 90%:** strong match\n- **60 to 90%:** probably right, check the runner-up\n"
                    "- **Below 60%:** uncertain, try another photo")

with photo_tab:
    st.caption("The model was trained on lab photos of single leaves on plain backgrounds. Photos like that work best.")
    for col, check in zip(st.columns(len(checks)), checks):
        with col.container(border=True):
            st.metric(f"{'✅' if check.ok else '⚠️'} {check.name}", check.value)
            st.caption("Looks good." if check.ok else check.tip)

st.download_button(
    "📄 Download report",
    build_report(source_name, results, info, checks, notes),
    file_name=f"plantdoc_{top_name.lower()}.md",
    mime="text/markdown",
)

st.divider()
st.caption(
    "Disclaimer: predictions are model estimates based on lab photos from the PlantVillage dataset. "
    "They should not replace a diagnosis from an agricultural expert or plant pathologist."
)
