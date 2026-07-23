from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from class_names import class_label
from config import (
    LOW_CONFIDENCE_THRESHOLD,
    MODEL_CONFIGS,
    MODEL_ORDER,
    SAMPLES_DIR,
)
from gradcam import (
    GradCAMModel,
    find_working_gradcam_model,
    generate_gradcam,
)
from image_processing import (
    ImageValidationError,
    image_fingerprint,
    image_to_array_255,
    load_image_from_bytes,
    load_image_from_path,
    prepare_model_input,
)
from metrics_loader import (
    EvidenceTables,
    load_evidence_tables,
    recommended_model,
)
from model_loader import ModelBundle, load_model_bundle_uncached
from prediction import PredictionResult, predict_top_k
from styles import APP_CSS


st.set_page_config(
    page_title="Explainable Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(APP_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model_bundle(model_name: str) -> ModelBundle:
    # Cache the loaded TensorFlow model across Streamlit reruns.
    return load_model_bundle_uncached(model_name)


@st.cache_data(show_spinner=False)
def load_tables() -> EvidenceTables:
    # Cache small CSV experiment summaries.
    return load_evidence_tables()


def format_percent(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def sample_files() -> list[Path]:
    # Return supported sample images bundled with the app.
    extensions = {".png", ".jpg", ".jpeg", ".ppm"}
    return sorted(
        path
        for path in SAMPLES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def render_top_predictions(result: PredictionResult) -> None:
    st.subheader("Top-3 predictions")

    for item in result.top_predictions:
        label_column, value_column = st.columns([4, 1])

        with label_column:
            st.markdown(
                f"<div class='rank-label'>#{item.rank} · "
                f"Class {item.class_id}: {item.class_name}</div>",
                unsafe_allow_html=True,
            )
            st.progress(float(item.confidence))

        with value_column:
            st.metric(
                "Confidence",
                format_percent(item.confidence),
                label_visibility="collapsed",
            )


def model_metric_row(
    tables: EvidenceTables,
    model_name: str,
) -> dict[str, Any]:
    comparison = tables.comparison[
        tables.comparison["model"] == model_name
    ]
    robustness = tables.robustness[
        tables.robustness["model"] == model_name
    ]
    gradcam = tables.gradcam[
        tables.gradcam["model"] == model_name
    ]
    shap = tables.shap[
        tables.shap["model"] == model_name
    ]

    return {
        "accuracy": (
            float(comparison.iloc[0]["accuracy"])
            if not comparison.empty
            else None
        ),
        "macro_f1": (
            float(comparison.iloc[0]["macro_f1"])
            if not comparison.empty
            else None
        ),
        "model_size_mb": (
            float(comparison.iloc[0]["model_file_size_mb"])
            if not comparison.empty
            else None
        ),
        "corrupted_accuracy": (
            float(robustness.iloc[0]["average_corrupted_accuracy"])
            if not robustness.empty
            else None
        ),
        "gradcam_roi": (
            float(gradcam.iloc[0]["mean_roi_attention_ratio"])
            if not gradcam.empty
            else None
        ),
        "shap_roi": (
            float(
                shap.iloc[0][
                    "mean_roi_absolute_attribution_ratio"
                ]
            )
            if not shap.empty
            else None
        ),
    }


def render_model_summary(
    tables: EvidenceTables,
    model_name: str,
) -> None:
    metrics = model_metric_row(tables, model_name)

    columns = st.columns(4)
    columns[0].metric(
        "Test accuracy",
        format_percent(metrics["accuracy"]),
    )
    columns[1].metric(
        "Macro F1",
        format_percent(metrics["macro_f1"]),
    )
    columns[2].metric(
        "Corrupted accuracy",
        format_percent(metrics["corrupted_accuracy"]),
    )
    columns[3].metric(
        "Model size",
        f"{metrics['model_size_mb']:.1f} MB",
    )

    st.caption(MODEL_CONFIGS[model_name].description)


def prepare_selected_image(
    source_mode: str,
) -> tuple[Image.Image | None, bytes | None, str | None]:
    if source_mode == "Upload an image":
        uploaded_file = st.file_uploader(
            "Upload a traffic-sign image",
            type=["png", "jpg", "jpeg", "ppm"],
            accept_multiple_files=False,
            help="Supported formats: PNG, JPG, JPEG, and PPM.",
        )

        if uploaded_file is None:
            return None, None, None

        image_bytes = uploaded_file.getvalue()

        try:
            image = load_image_from_bytes(image_bytes)
        except ImageValidationError as error:
            st.error(str(error))
            return None, None, None

        return image, image_bytes, uploaded_file.name

    available_samples = sample_files()

    if not available_samples:
        st.warning("No sample images were found in the samples folder.")
        return None, None, None

    selected_name = st.selectbox(
        "Choose a bundled sample",
        options=[path.name for path in available_samples],
    )

    selected_path = SAMPLES_DIR / selected_name
    image, image_bytes = load_image_from_path(selected_path)

    return image, image_bytes, selected_name


def get_or_create_gradcam_model(
    model_name: str,
    bundle: ModelBundle,
    model_input: np.ndarray,
    class_index: int,
) -> GradCAMModel:
    # Keep one validated Grad-CAM graph per model in the current browser session.
    if "gradcam_models" not in st.session_state:
        st.session_state["gradcam_models"] = {}

    cache: dict[str, GradCAMModel] = st.session_state[
        "gradcam_models"
    ]

    if model_name not in cache:
        cache[model_name] = find_working_gradcam_model(
            bundle.model,
            model_input,
            class_index,
        )

    return cache[model_name]


def render_classifier_page(
    tables: EvidenceTables,
    default_model: str,
) -> None:
    st.title("Explainable Traffic Sign Recognition")
    st.markdown(
        "<div class='app-subtitle'>Classify a GTSRB traffic sign with "
        "Custom CNN, EfficientNetV2B0, or ConvNeXtTiny and inspect a "
        "Grad-CAM explanation.</div>",
        unsafe_allow_html=True,
    )

    selected_model = st.selectbox(
        "Model",
        options=MODEL_ORDER,
        index=MODEL_ORDER.index(default_model),
        help=(
            "ConvNeXtTiny is selected by default because it achieved the "
            "strongest combined clean and corrupted-test performance."
        ),
    )

    render_model_summary(
        tables,
        selected_model,
    )

    st.divider()

    source_mode = st.radio(
        "Image source",
        options=[
            "Upload an image",
            "Use a bundled sample",
        ],
        horizontal=True,
    )

    image, image_bytes, image_name = prepare_selected_image(
        source_mode
    )

    if image is None or image_bytes is None:
        st.info(
            "Upload an image or choose a bundled sample to run the classifier."
        )
        return

    image_255 = image_to_array_255(image)

    with st.spinner(f"Loading {selected_model}..."):
        bundle = load_model_bundle(selected_model)

    model_input = prepare_model_input(
        image_255,
        bundle.input_range,
    )

    analysis_key = (
        selected_model,
        image_fingerprint(image_bytes),
    )

    if (
        "prediction_cache" not in st.session_state
        or st.session_state.get("prediction_key") != analysis_key
    ):
        with st.spinner("Running classification..."):
            prediction = predict_top_k(
                bundle,
                model_input,
                top_k=3,
            )

        st.session_state["prediction_key"] = analysis_key
        st.session_state["prediction_cache"] = prediction
    else:
        prediction = st.session_state["prediction_cache"]

    image_column, result_column = st.columns([1, 1.15])

    with image_column:
        st.subheader("Input image")
        st.image(
            image,
            caption=image_name,
            use_container_width=True,
        )
        st.caption(
            f"Original size: {image.width} × {image.height} pixels · "
            "Model input: 224 × 224 RGB"
        )

    with result_column:
        st.subheader("Prediction")
        st.markdown(
            "<div class='prediction-card'>"
            f"<div class='prediction-title'>Class "
            f"{prediction.predicted_class_id}: "
            f"{prediction.predicted_class_name}</div>"
            f"<div class='prediction-meta'>Confidence: "
            f"{format_percent(prediction.confidence)} · "
            f"Inference: {prediction.inference_ms:.1f} ms</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        if prediction.confidence < LOW_CONFIDENCE_THRESHOLD:
            st.warning(
                "Low-confidence prediction. Review the Top-3 classes and "
                "Grad-CAM explanation before interpreting the result."
            )
        else:
            st.success("The model produced a high-confidence prediction.")

        render_top_predictions(prediction)

    st.divider()
    st.subheader("Grad-CAM explanation")
    st.caption(
        "Warm regions contribute more strongly to the selected predicted "
        "class. Grad-CAM is an explanatory diagnostic, not a proof that the "
        "model reasoned correctly."
    )

    explanation_key = analysis_key

    if st.button(
        "Generate Grad-CAM",
        type="primary",
        use_container_width=False,
    ):
        with st.spinner(
            "Finding a valid spatial layer and creating Grad-CAM..."
        ):
            gradcam_model = get_or_create_gradcam_model(
                selected_model,
                bundle,
                model_input,
                prediction.predicted_class_id,
            )

            explanation = generate_gradcam(
                gradcam_model,
                model_input,
                image_255,
                prediction.predicted_class_id,
            )

        if "gradcam_results" not in st.session_state:
            st.session_state["gradcam_results"] = {}

        st.session_state["gradcam_results"][
            explanation_key
        ] = explanation

    explanation = st.session_state.get(
        "gradcam_results",
        {},
    ).get(explanation_key)

    if explanation is None:
        st.info(
            "Press Generate Grad-CAM to create the explanation for this "
            "image and selected model."
        )
        return

    original_column, heatmap_column, overlay_column = st.columns(3)

    with original_column:
        st.image(
            image_255.astype(np.uint8),
            caption="Resized model input",
            use_container_width=True,
        )

    with heatmap_column:
        st.image(
            explanation.heatmap,
            caption="Grad-CAM heatmap",
            clamp=True,
            use_container_width=True,
        )

    with overlay_column:
        st.image(
            explanation.overlay,
            caption="Grad-CAM overlay",
            use_container_width=True,
        )

    st.caption(
        f"Target feature layer: `{explanation.target_layer_name}`"
    )


def render_evidence_page(
    tables: EvidenceTables,
    default_model: str,
) -> None:
    st.title("Model evidence")
    st.write(
        "The deployment recommendation uses the completed clean-test, "
        "robustness, Grad-CAM, and SHAP experiments."
    )

    st.success(
        f"Recommended default model: **{default_model}**"
    )

    clean_table = tables.comparison[
        [
            "model",
            "accuracy",
            "macro_f1",
            "top3_accuracy",
            "parameter_count",
            "model_file_size_mb",
            "milliseconds_per_image",
        ]
    ].copy()

    for column in [
        "accuracy",
        "macro_f1",
        "top3_accuracy",
    ]:
        clean_table[column] = (
            clean_table[column] * 100.0
        ).round(2)

    clean_table = clean_table.rename(
        columns={
            "model": "Model",
            "accuracy": "Accuracy (%)",
            "macro_f1": "Macro F1 (%)",
            "top3_accuracy": "Top-3 (%)",
            "parameter_count": "Parameters",
            "model_file_size_mb": "Size (MB)",
            "milliseconds_per_image": "Recorded ms/image",
        }
    )

    st.subheader("Clean test comparison")
    st.dataframe(
        clean_table,
        hide_index=True,
        use_container_width=True,
    )

    robustness_table = tables.robustness.copy()

    for column in robustness_table.columns:
        if column != "model":
            robustness_table[column] = (
                robustness_table[column] * 100.0
            ).round(2)

    robustness_table = robustness_table.rename(
        columns={
            "model": "Model",
            "clean_accuracy": "Clean accuracy (%)",
            "average_corrupted_accuracy": "Average corrupted accuracy (%)",
            "average_accuracy_drop": "Average accuracy drop (points)",
            "worst_case_accuracy": "Worst-case accuracy (%)",
        }
    )

    st.subheader("Robustness summary")
    st.dataframe(
        robustness_table,
        hide_index=True,
        use_container_width=True,
    )

    explanation_table = tables.gradcam[
        [
            "model",
            "mean_roi_attention_ratio",
            "median_roi_attention_ratio",
        ]
    ].merge(
        tables.shap[
            [
                "model",
                "mean_roi_absolute_attribution_ratio",
                "mean_shap_seconds",
            ]
        ],
        on="model",
        how="outer",
    )

    explanation_table = explanation_table.rename(
        columns={
            "model": "Model",
            "mean_roi_attention_ratio": "Mean Grad-CAM ROI ratio",
            "median_roi_attention_ratio": "Median Grad-CAM ROI ratio",
            "mean_roi_absolute_attribution_ratio": "Mean SHAP ROI ratio",
            "mean_shap_seconds": "Mean SHAP time (s)",
        }
    )

    st.subheader("Explainability summary")
    st.dataframe(
        explanation_table,
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        "ROI ratios are diagnostic measures from small representative "
        "explanation samples. They should be interpreted together with the "
        "saved Grad-CAM and SHAP figures, not as standalone accuracy metrics."
    )


def render_about_page() -> None:
    st.title("About this application")
    st.markdown(
        """
This application is the deployment component of the project **Explainable
Traffic Sign Recognition Using Custom CNN, EfficientNetV2, and ConvNeXtTiny**.

It supports:

- PNG, JPG, JPEG, and PPM image input;
- selection among all three trained classifiers;
- predicted GTSRB class and confidence;
- Top-3 alternatives;
- live Grad-CAM heatmaps and overlays;
- experiment evidence used to select the default model.

### Important limitation

This is a research and educational prototype. It is not validated for vehicle
control, navigation, legal interpretation, or safety-critical road decisions.
Predictions can fail under domain shift, unusual countries or sign systems,
severe image degradation, partial signs, or images outside the 43 GTSRB
classes.
        """
    )


try:
    evidence_tables = load_tables()
    default_model_name = recommended_model(evidence_tables)
except Exception as error:
    st.error(f"Could not load experiment summaries: {error}")
    st.stop()


with st.sidebar:
    st.header("Navigation")

    page = st.radio(
        "Page",
        options=[
            "Classifier",
            "Model evidence",
            "About",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Recommended model**")
    st.write(default_model_name)
    st.caption(
        "Selected from clean accuracy and average corrupted-test accuracy."
    )

    st.divider()
    st.caption(
        "Research prototype · 43 GTSRB classes · 224 × 224 RGB input"
    )


if page == "Classifier":
    render_classifier_page(
        evidence_tables,
        default_model_name,
    )
elif page == "Model evidence":
    render_evidence_page(
        evidence_tables,
        default_model_name,
    )
else:
    render_about_page()
