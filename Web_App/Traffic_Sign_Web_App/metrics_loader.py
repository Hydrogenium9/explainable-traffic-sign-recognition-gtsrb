from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import RESULT_FILES


@dataclass(frozen=True)
class EvidenceTables:
    comparison: pd.DataFrame
    robustness: pd.DataFrame
    gradcam: pd.DataFrame
    shap: pd.DataFrame


def load_evidence_tables() -> EvidenceTables:
    # Load the experiment summaries bundled with the application.
    missing = [
        str(path)
        for path in RESULT_FILES.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing experiment result files:\n" + "\n".join(missing)
        )

    return EvidenceTables(
        comparison=pd.read_csv(RESULT_FILES["comparison"]),
        robustness=pd.read_csv(RESULT_FILES["robustness"]),
        gradcam=pd.read_csv(RESULT_FILES["gradcam"]),
        shap=pd.read_csv(RESULT_FILES["shap"]),
    )


def recommended_model(tables: EvidenceTables) -> str:
    # Select the strongest model using clean accuracy and corrupted accuracy.
    clean = tables.comparison[["model", "accuracy"]].copy()
    robust = tables.robustness[
        ["model", "average_corrupted_accuracy"]
    ].copy()

    merged = clean.merge(robust, on="model", how="inner")

    if merged.empty:
        return str(
            tables.comparison.sort_values(
                "accuracy",
                ascending=False,
            ).iloc[0]["model"]
        )

    merged["deployment_score"] = (
        merged["accuracy"] * 0.60
        + merged["average_corrupted_accuracy"] * 0.40
    )

    return str(
        merged.sort_values(
            "deployment_score",
            ascending=False,
        ).iloc[0]["model"]
    )
