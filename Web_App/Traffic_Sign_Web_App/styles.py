from __future__ import annotations


APP_CSS = """
<style>
    .main .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .app-subtitle {
        color: #5f6b7a;
        font-size: 1.02rem;
        margin-top: -0.7rem;
        margin-bottom: 1.4rem;
    }

    .prediction-card {
        border: 1px solid rgba(49, 51, 63, 0.18);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        background: rgba(250, 250, 250, 0.65);
        margin-bottom: 0.8rem;
    }

    .prediction-title {
        font-size: 1.35rem;
        font-weight: 700;
        line-height: 1.3;
    }

    .prediction-meta {
        color: #5f6b7a;
        margin-top: 0.25rem;
    }

    .rank-label {
        font-weight: 600;
        margin-bottom: 0.1rem;
    }

    .small-note {
        color: #687386;
        font-size: 0.9rem;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.45rem;
    }
</style>
"""
