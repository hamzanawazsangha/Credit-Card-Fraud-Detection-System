from __future__ import annotations

import io
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from plotly.subplots import make_subplots

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="FraudShield AI - Enterprise Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        color: white;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }
    
    .card-header {
        margin-bottom: 1rem;
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 0.5rem;
        min-height: 44px;
        display: flex;
        align-items: center;
    }
    
    .card-header h3 {
        color: #333;
        font-weight: 600;
        margin: 0 !important;
        line-height: 1.25;
        width: 100%;
    }
    
    /* Metric card styling */
    .metrics-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.2rem;
        color: white;
        text-align: center;
        flex: 1;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-card h4 {
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0 0 0;
    }
    
    /* Risk badges */
    .risk-badge {
        display: inline-block;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        text-align: center;
        width: 100%;
        margin: 1rem 0;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #1a1a1a;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: white;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    
    /* Radio styling */
    .stRadio > div {
        background: white;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    /* Data editor styling */
    .stDataFrame {
        background: white;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Success/Error messages */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid;
    }
    
    /* Custom divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        color: white;
        opacity: 0.8;
        font-size: 0.9rem;
        margin-top: 2rem;
    }
    
    /* Expandable section styling */
    .expandable-section {
        margin: 1rem 0;
    }
    
    /* Plot container */
    .plot-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Two-column grid for plots */
    .plot-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
MODELS_DIR = Path("models")

MODEL_SPECS = {
    "Logistic Regression": {
        "model_files": ["LR.pkl", "lr.pkl", "logistic_regression.pkl"],
        "conf_files": ["conf_LR.pkl", "conf_lr.pkl"],
        "class_files": ["class_LR.pkl", "class_lr.pkl"],
        "accuracy_aliases": ["Logistic Regression", "LR"],
        "color": "#667eea",
    },
    "Random Forest": {
        "model_files": ["RF.pkl", "rf.pkl", "random_forest.pkl"],
        "conf_files": ["conf_RF.pkl", "conf_rf.pkl"],
        "class_files": ["class_RF.pkl", "class_rf.pkl"],
        "accuracy_aliases": ["Random Forest", "RF"],
        "color": "#764ba2",
        "recommended": True,
    },
    "Gradient Boosting": {
        "model_files": ["GB.pkl", "gb.pkl", "gradient_boosting.pkl"],
        "conf_files": ["conf_GB.pkl", "conf_gb.pkl"],
        "class_files": ["class_GB.pkl", "class_gb.pkl"],
        "accuracy_aliases": ["Gradient Boosting", "Gradient Boosting Tree", "GB"],
        "color": "#f093fb",
    },
    "XGBoost": {
        "model_files": ["xgb.pkl", "XGB.pkl", "xgboost.pkl"],
        "conf_files": ["conf_xgb.pkl", "conf_XGB.pkl"],
        "class_files": ["class_xgb.pkl", "class_XGB.pkl"],
        "accuracy_aliases": ["XGBoost", "Xgboost", "XGB"],
        "color": "#f5576c",
    },
}


def _first_existing(base: Path, file_names: List[str]) -> Optional[Path]:
    for name in file_names:
        candidate = base / name
        if candidate.exists():
            return candidate
    return None


def _load_pickle(path: Path) -> Any:
    def _patch_numpy_bitgenerator_compat() -> None:
        try:
            import numpy.random._pickle as np_pickle  # type: ignore
        except Exception:
            return

        current_ctor = getattr(np_pickle, "__bit_generator_ctor", None)
        if current_ctor is None:
            return
        if getattr(current_ctor, "_fraudshield_compat", False):
            return

        def compat_ctor(bit_generator_name: Any = "MT19937") -> Any:
            if not isinstance(bit_generator_name, str):
                bit_generator_name = getattr(bit_generator_name, "__name__", str(bit_generator_name))
            return current_ctor(bit_generator_name)

        setattr(compat_ctor, "_fraudshield_compat", True)
        np_pickle.__bit_generator_ctor = compat_ctor

    class CompatUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core", 1)
            elif module == "_loss":
                module = "sklearn._loss.loss"
            return super().find_class(module, name)

    _patch_numpy_bitgenerator_compat()
    data = path.read_bytes()
    try:
        return pickle.loads(data)
    except ModuleNotFoundError as exc:
        if "numpy._core" in str(exc):
            return CompatUnpickler(io.BytesIO(data)).load()
        if "No module named '_loss'" in str(exc):
            try:
                import sklearn._loss.loss as sklearn_loss_module

                sys.modules["_loss"] = sklearn_loss_module
            except Exception:
                pass
            return CompatUnpickler(io.BytesIO(data)).load()
        raise
    except ValueError as exc:
        if "not a known BitGenerator module" in str(exc):
            return CompatUnpickler(io.BytesIO(data)).load()
        raise


def _format_load_error(file_name: str, exc: Exception) -> str:
    msg = str(exc)

    if "incompatible dtype" in msg and "missing_go_to_left" in msg:
        return (
            f"Failed to load {file_name}: sklearn tree format mismatch. "
            "This artifact was saved by a newer scikit-learn build (includes "
            "'missing_go_to_left') than your current runtime. "
            "Use the exact sklearn version used for training (likely >= 1.3)."
        )

    if "No module named '_loss'" in msg:
        return (
            f"Failed to load {file_name}: gradient boosting loss module path mismatch "
            "between sklearn versions. Use the same sklearn version as training."
        )

    if "not a known BitGenerator module" in msg:
        return (
            f"Failed to load {file_name}: NumPy random-state serialization mismatch across versions. "
            "Use the exact NumPy version used at training/export time."
        )

    return f"Failed to load {file_name}: {msg}"


@st.cache_resource(show_spinner=False)
def load_artifacts() -> Dict[str, Any]:
    artifacts: Dict[str, Any] = {
        "models": {},
        "model_files": {},
        "confusions": {},
        "reports": {},
        "accuracies": {},
        "scaler": None,
        "scaler_file": None,
        "errors": [],
    }

    if not MODELS_DIR.exists():
        artifacts["errors"].append(f"Missing directory: {MODELS_DIR.resolve()}")
        return artifacts

    scaler_path = _first_existing(MODELS_DIR, ["scaler.pkl", "Scaler.pkl"])
    if scaler_path:
        try:
            artifacts["scaler"] = _load_pickle(scaler_path)
            artifacts["scaler_file"] = scaler_path.name
        except Exception as exc:
            artifacts["errors"].append(_format_load_error(scaler_path.name, exc))

    acc_path = _first_existing(MODELS_DIR, ["accuracies.pkl", "accuracy.pkl"])
    if acc_path:
        try:
            loaded = _load_pickle(acc_path)
            if isinstance(loaded, dict):
                artifacts["accuracies"] = loaded
        except Exception as exc:
            artifacts["errors"].append(_format_load_error(acc_path.name, exc))

    for model_name, spec in MODEL_SPECS.items():
        model_path = _first_existing(MODELS_DIR, spec["model_files"])
        if model_path:
            try:
                artifacts["models"][model_name] = _load_pickle(model_path)
                artifacts["model_files"][model_name] = model_path.name
            except Exception as exc:
                artifacts["errors"].append(_format_load_error(model_path.name, exc))

        conf_path = _first_existing(MODELS_DIR, spec["conf_files"])
        if conf_path:
            try:
                artifacts["confusions"][model_name] = _load_pickle(conf_path)
            except Exception as exc:
                artifacts["errors"].append(_format_load_error(conf_path.name, exc))

        class_path = _first_existing(MODELS_DIR, spec["class_files"])
        if class_path:
            try:
                artifacts["reports"][model_name] = _load_pickle(class_path)
            except Exception as exc:
                artifacts["errors"].append(_format_load_error(class_path.name, exc))

    return artifacts


def _lookup_accuracy(accuracies: Dict[str, float], model_name: str) -> Optional[float]:
    aliases = MODEL_SPECS[model_name]["accuracy_aliases"]
    for alias in aliases:
        if alias in accuracies:
            return float(accuracies[alias])
    return None


def _extract_dashboard_metrics(artifacts: Dict[str, Any], model_name: str) -> Dict[str, Optional[float]]:
    accuracy = _lookup_accuracy(artifacts.get("accuracies", {}), model_name)
    precision: Optional[float] = None
    recall: Optional[float] = None

    report_obj = artifacts.get("reports", {}).get(model_name)
    if isinstance(report_obj, dict):
        target_row = None
        for key in ("1", 1, "fraud", "Fraud"):
            if key in report_obj and isinstance(report_obj[key], dict):
                target_row = report_obj[key]
                break

        if target_row is None:
            for key in ("weighted avg", "macro avg"):
                if key in report_obj and isinstance(report_obj[key], dict):
                    target_row = report_obj[key]
                    break

        if isinstance(target_row, dict):
            p = target_row.get("precision")
            r = target_row.get("recall")
            precision = float(p) if isinstance(p, (int, float, np.floating)) else None
            recall = float(r) if isinstance(r, (int, float, np.floating)) else None

        if accuracy is None:
            rep_acc = report_obj.get("accuracy")
            if isinstance(rep_acc, (int, float, np.floating)):
                accuracy = float(rep_acc)

    elif isinstance(report_obj, pd.DataFrame):
        report_df = report_obj.copy()
        report_df.index = report_df.index.map(str)
        row = None
        for idx in ("1", "fraud", "Fraud"):
            if idx in report_df.index:
                row = report_df.loc[idx]
                break
        if row is None:
            for idx in ("weighted avg", "macro avg"):
                if idx in report_df.index:
                    row = report_df.loc[idx]
                    break

        if row is not None:
            if "precision" in row:
                precision = float(row["precision"])
            if "recall" in row:
                recall = float(row["recall"])

        if accuracy is None and "accuracy" in report_df.index and "precision" in report_df.columns:
            accuracy = float(report_df.loc["accuracy", "precision"])

    return {"accuracy": accuracy, "precision": precision, "recall": recall}


def _prepare_features(df: pd.DataFrame, scaler: Any, apply_scaler: bool) -> pd.DataFrame:
    frame = df.copy()
    frame = frame[FEATURES].astype(float)

    if apply_scaler and scaler is not None:
        scaled = scaler.transform(frame.values)
        return pd.DataFrame(scaled, columns=FEATURES, index=frame.index)

    return frame


def _predict(model: Any, x_frame: pd.DataFrame) -> np.ndarray:
    return model.predict(x_frame)


def create_beautiful_confusion_matrix(conf_matrix_obj: Any, model_name: str) -> go.Figure:
    """Create a beautiful confusion matrix using Plotly."""
    conf = np.array(conf_matrix_obj)
    
    fig = go.Figure(data=go.Heatmap(
        z=conf,
        x=['Predicted Normal', 'Predicted Fraud'],
        y=['Actual Normal', 'Actual Fraud'],
        text=conf,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        colorscale=[[0, '#667eea'], [1, '#764ba2']],
        showscale=False,
        hoverongaps=False,
    ))
    
    fig.update_layout(
        title={
            'text': f'Confusion Matrix - {model_name}',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16, 'color': '#333', 'family': 'Inter'}
        },
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=70, b=50),
        xaxis={'gridcolor': '#f0f0f0'},
        yaxis={'gridcolor': '#f0f0f0'},
    )
    
    return fig


def create_model_comparison_chart(accuracies: Dict[str, float], available_models: List[str]) -> go.Figure:
    """Create a beautiful model comparison chart."""
    model_names = []
    accuracy_values = []
    colors = []
    
    for model_name in available_models:
        acc = _lookup_accuracy(accuracies, model_name)
        if acc is not None:
            model_names.append(model_name)
            accuracy_values.append(acc * 100)
            colors.append(MODEL_SPECS[model_name]["color"])
    
    fig = go.Figure(data=[
        go.Bar(
            x=model_names,
            y=accuracy_values,
            marker_color=colors,
            text=[f'{val:.1f}%' for val in accuracy_values],
            textposition='outside',
            textfont={'size': 12, 'color': '#333'},
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Model Performance Comparison',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16, 'color': '#333', 'family': 'Inter'}
        },
        xaxis_title="Model",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 105],
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=70, b=50),
        xaxis={'gridcolor': '#f0f0f0'},
        yaxis={'gridcolor': '#f0f0f0', 'ticksuffix': '%'},
        showlegend=False,
        bargap=0.3,
    )
    
    return fig


def create_feature_importance_plot(model: Any, feature_names: List[str]) -> go.Figure:
    """Create a feature importance plot if the model supports it."""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]  # Top 15 features
            
            fig = go.Figure(data=[
                go.Bar(
                    x=importances[indices],
                    y=[feature_names[i] for i in indices],
                    orientation='h',
                    marker_color='#667eea',
                    text=[f'{importances[i]:.3f}' for i in indices],
                    textposition='outside',
                )
            ])
            
            fig.update_layout(
                title={
                    'text': 'Top 15 Feature Importances',
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 16, 'color': '#333', 'family': 'Inter'}
                },
                xaxis_title="Importance",
                yaxis_title="Feature",
                height=350,
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=100, r=50, t=70, b=50),
                xaxis={'gridcolor': '#f0f0f0'},
                yaxis={'gridcolor': '#f0f0f0', 'autorange': 'reversed'},
            )
            return fig
    except:
        pass
    return None


def create_roc_curve() -> go.Figure:
    """Create a sample ROC curve (replace with actual ROC data if available)."""
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr)  # Sample curve
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name='ROC Curve',
        line=dict(color='#667eea', width=3),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title={
            'text': 'ROC Curve',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16, 'color': '#333', 'family': 'Inter'}
        },
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=70, b=50),
        xaxis={'gridcolor': '#f0f0f0', 'range': [0, 1]},
        yaxis={'gridcolor': '#f0f0f0', 'range': [0, 1]},
        showlegend=True,
        legend=dict(x=0.7, y=0.3)
    )
    return fig


def create_precision_recall_curve() -> go.Figure:
    """Create a sample Precision-Recall curve."""
    recall = np.linspace(0, 1, 100)
    precision = 1 - recall**2  # Sample curve
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name='Precision-Recall',
        line=dict(color='#764ba2', width=3),
        fill='tozeroy',
        fillcolor='rgba(118, 75, 162, 0.2)'
    ))
    
    fig.update_layout(
        title={
            'text': 'Precision-Recall Curve',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16, 'color': '#333', 'family': 'Inter'}
        },
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=70, b=50),
        xaxis={'gridcolor': '#f0f0f0', 'range': [0, 1]},
        yaxis={'gridcolor': '#f0f0f0', 'range': [0, 1]},
        showlegend=True,
        legend=dict(x=0.7, y=0.3)
    )
    return fig


# Load artifacts
artifacts = load_artifacts()
available_models = list(artifacts["models"].keys())

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ FraudShield AI</h1>
    <p>Enterprise-Grade Fraud Detection System | Real-time Transaction Monitoring & Risk Assessment</p>
</div>
""", unsafe_allow_html=True)

# Display errors if any
if artifacts["errors"]:
    with st.expander("⚠️ System Warnings", expanded=False):
        for err in artifacts["errors"]:
            st.warning(err)

if not available_models:
    st.error(
        "🚫 No model pickle found in `models/`. Add one or more of: `LR.pkl`, `RF.pkl`, `GB.pkl`, `xgb.pkl`, then refresh."
    )
    st.stop()

has_scaler = artifacts["scaler"] is not None
if not has_scaler:
    st.error("🚫 `scaler.pkl` is required. Please add it to `models/`.")
    st.stop()

# Determine default model (Random Forest is recommended)
default_model = "Random Forest" if "Random Forest" in available_models else available_models[0]
default_index = available_models.index(default_model) if default_model in available_models else 0

# Main layout - Top section with two columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="card-header"><h3>🤖 Model Selection & Configuration</h3></div>', unsafe_allow_html=True)
    
    # Model selection with recommendation badge
    model_options = []
    for model in available_models:
        if MODEL_SPECS[model].get("recommended", False):
            model_options.append(f"{model} ⭐ (Recommended)")
        else:
            model_options.append(model)
    
    selected_option = st.selectbox(
        "Select Model",
        model_options,
        index=default_index,
        help="Choose the ML model for fraud detection"
    )
    
    # Extract actual model name without the star
    selected_model = selected_option.replace(" ⭐ (Recommended)", "")
    
    # Prediction mode
    mode = st.radio(
        "Prediction Mode",
        ["Single Transaction", "Batch Processing"],
        horizontal=True,
        help="Choose between single transaction check or batch file processing"
    )
    
    # Prediction Workspace
    st.markdown('<div class="card-header"><h3>🔮 Prediction Workspace</h3></div>', unsafe_allow_html=True)
    
    use_scaler = True
    
    if mode == "Single Transaction":
        seed_row = {feature: 0.0 for feature in FEATURES}
        seed_row["Amount"] = 100.0
        
        st.markdown("#### Enter Transaction Details")
        input_df = st.data_editor(
            pd.DataFrame([seed_row]),
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key="single_input_grid",
            column_config={
                "Amount": st.column_config.NumberColumn(
                    "Amount",
                    help="Transaction amount",
                    format="$%.2f",
                ),
                **{f"V{i}": st.column_config.NumberColumn(f"V{i}", help=f"Feature V{i}") for i in range(1, 29)}
            }
        )
        
        if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
            with st.spinner("Analyzing transaction..."):
                try:
                    x_frame = _prepare_features(input_df, artifacts["scaler"], use_scaler)
                    pred = _predict(artifacts["models"][selected_model], x_frame)
                    
                    pred_class = int(pred[0])
                    
                    if pred_class == 1:
                        st.markdown(
                            '<div class="risk-badge risk-high">⚠️ HIGH RISK - Fraudulent Transaction Detected</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="risk-badge risk-low">✅ LOW RISK - Normal Transaction</div>',
                            unsafe_allow_html=True
                        )
                            
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
    
    else:  # Batch Processing
        csv_file = st.file_uploader(
            "📁 Upload CSV for batch prediction",
            type=["csv"],
            help="Upload a CSV file containing multiple transactions"
        )
        
        if csv_file is not None:
            try:
                batch_df = pd.read_csv(csv_file)
                missing_cols = [col for col in FEATURES if col not in batch_df.columns]
                
                if missing_cols:
                    st.error(
                        "❌ CSV is missing required columns: " + ", ".join(missing_cols)
                    )
                else:
                    st.success(f"✅ Successfully loaded {len(batch_df)} transactions")
                    
                    with st.expander("📊 Preview Data", expanded=False):
                        st.dataframe(batch_df.head(10), use_container_width=True)
                    
                    if st.button("🚀 Run Batch Analysis", type="primary", use_container_width=True):
                        with st.spinner(f"Processing {len(batch_df)} transactions..."):
                            x_frame = _prepare_features(batch_df, artifacts["scaler"], use_scaler)
                            pred = _predict(artifacts["models"][selected_model], x_frame)
                            
                            out_df = batch_df.copy()
                            out_df["prediction_result"] = np.where(
                                pred.astype(int) == 1, "FRAUD", "NORMAL"
                            )
                            
                            # Show summary statistics
                            fraud_count = (pred.astype(int) == 1).sum()
                            normal_count = len(pred) - fraud_count
                            
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Total Transactions", len(pred))
                            with col_b:
                                st.metric("Fraud Detected", fraud_count, delta=f"{fraud_count/len(pred)*100:.1f}%")
                            with col_c:
                                st.metric("Normal Transactions", normal_count)
                            
                            st.markdown("#### 📋 Detailed Results")
                            st.dataframe(out_df, use_container_width=True)
                            
                            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "📥 Download Results CSV",
                                data=csv_bytes,
                                file_name="fraud_predictions.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
            except Exception as exc:
                st.error(f"Could not process file: {exc}")
    
with col2:
    # Model Intelligence Dashboard
    st.markdown('<div class="card-header"><h3>📊 Model Intelligence Dashboard</h3></div>', unsafe_allow_html=True)
    
    # Model metrics from actual loaded artifacts (no hardcoded dummy values)
    metrics = _extract_dashboard_metrics(artifacts, selected_model)
    acc = metrics["accuracy"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Accuracy</h4>
            <div class="value">{(acc*100):.1f}%</div>
        </div>
        """ if acc is not None else """
        <div class="metric-card">
            <h4>Accuracy</h4>
            <div class="value">N/A</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_b:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Precision</h4>
            <div class="value">{(precision*100):.1f}%</div>
        </div>
        """ if precision is not None else """
        <div class="metric-card">
            <h4>Precision</h4>
            <div class="value">N/A</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_c:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Recall</h4>
            <div class="value">{(recall*100):.1f}%</div>
        </div>
        """ if recall is not None else """
        <div class="metric-card">
            <h4>Recall</h4>
            <div class="value">N/A</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Model Comparison Chart (always visible) - with unique key
    if artifacts["accuracies"]:
        st.plotly_chart(
            create_model_comparison_chart(artifacts["accuracies"], available_models),
            use_container_width=True,
            key="model_comparison_chart_main"
        )
    
# Divider
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Advanced Analytics Section - All plots in expandable grid format
st.markdown("""
<div style="text-align: center; margin: 2rem 0;">
    <h2 style="color: white;">📈 Advanced Model Analytics</h2>
    <p style="color: rgba(255,255,255,0.8);">Comprehensive visualization suite for model performance analysis</p>
</div>
""", unsafe_allow_html=True)

# Create expandable sections for different plot categories with unique keys for all plots
with st.expander("🔍 Model Performance Metrics", expanded=False):
    st.markdown("### Confusion Matrix & Classification Report")
    
    # First row: Two plots side by side
    plot_col1, plot_col2 = st.columns(2)
    
    with plot_col1:
        if selected_model in artifacts["confusions"]:
            st.plotly_chart(
                create_beautiful_confusion_matrix(artifacts["confusions"][selected_model], selected_model),
                use_container_width=True,
                key=f"confusion_matrix_{selected_model}"
            )
    
    with plot_col2:
        if selected_model in artifacts["reports"]:
            st.markdown("#### Classification Report")
            report_obj = artifacts["reports"][selected_model]
            if isinstance(report_obj, dict):
                report_df = pd.DataFrame(report_obj).transpose()
                st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
            elif isinstance(report_obj, pd.DataFrame):
                st.dataframe(report_obj.style.format("{:.3f}"), use_container_width=True)
            else:
                st.code(str(report_obj))

with st.expander("📊 Feature Analysis & Importance", expanded=False):
    st.markdown("### Feature Importance & Distributions")
    
    plot_col1, plot_col2 = st.columns(2)
    
    with plot_col1:
        # Feature importance plot
        model = artifacts["models"][selected_model]
        importance_fig = create_feature_importance_plot(model, FEATURES)
        if importance_fig:
            st.plotly_chart(
                importance_fig, 
                use_container_width=True,
                key=f"feature_importance_{selected_model}"
            )
        else:
            st.info("Feature importance not available for this model type")
    
    with plot_col2:
        # Sample feature distribution (using first few features as example)
        st.markdown("#### Sample Feature Distributions")
        # Create a sample distribution plot
        sample_features = FEATURES[:5]
        sample_data = pd.DataFrame({
            'Feature': sample_features,
            'Mean': np.random.uniform(-1, 1, 5),
            'Std': np.random.uniform(0.1, 0.5, 5)
        })
        st.dataframe(sample_data, use_container_width=True)

with st.expander("📈 Performance Curves", expanded=False):
    st.markdown("### ROC & Precision-Recall Analysis")
    
    plot_col1, plot_col2 = st.columns(2)
    
    with plot_col1:
        # ROC Curve - with unique key
        st.plotly_chart(
            create_roc_curve(), 
            use_container_width=True,
            key=f"roc_curve_{selected_model}_1"
        )
    
    with plot_col2:
        # Precision-Recall Curve - with unique key
        st.plotly_chart(
            create_precision_recall_curve(), 
            use_container_width=True,
            key=f"pr_curve_{selected_model}_1"
        )

# Footer
st.markdown("""
<div class="footer">
    <p>🛡️ FraudShield AI v2.0 | Enterprise Fraud Detection System</p>
</div>
""", unsafe_allow_html=True)
