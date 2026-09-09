"""
Wi-Fi Indoor Localization and Signal Coverage Analysis
-------------------------------------------------------
Dataset  : UJIIndoorLoc (TrainingData.csv / ValidationData.csv)
Features : WAP001-WAP520 RSS fingerprints
Targets  : FLOOR classification + signal coverage mapping

Run with: streamlit run wifi_analysis.py
"""

import io
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from collections import deque
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)

# -------------------------------------------------------
# Page config and CSS
# -------------------------------------------------------
st.set_page_config(
    page_title="Wi-Fi Signal Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    /* Blue uppercase section titles */
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #4a9eff;
        border-bottom: 1px solid #2a2a3a;
        padding-bottom: 4px;
        margin-top: 1.4rem;
        margin-bottom: 0.4rem;
    }

    /* Plain-English explanation box under each section */
    .explain-box {
        background: #131a2a;
        border-left: 3px solid #4a9eff;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #c8d4e8;
        margin-bottom: 0.9rem;
        line-height: 1.6;
    }

    [data-testid="metric-container"] {
        background: #0e1117;
        border: 1px solid #1e2130;
        border-radius: 8px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Constants
# -------------------------------------------------------
MISSING_RAW    = 100     # how the dataset marks "router not detected"
MISSING_DBM    = -110    # what we replace it with (-110 dBm = very weak)
WEAK_THRESHOLD = -65     # signals below this are "weak zones"
                         # -60 dBm and above = good signal
                         # -70 dBm and below = poor signal
VAR_FILTER     = 1.5     # drop WAP columns that barely change across samples
GRID_BINS      = 30      # how fine the heatmap grid is
SMOOTH_SIGMA   = 1.2     # how much to smooth the heatmap

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
st.sidebar.header("Step 1 — Upload Data")
st.sidebar.caption("Upload the UJIIndoorLoc dataset files below.")
train_file = st.sidebar.file_uploader("TrainingData.csv",   type=["csv"], key="train")
val_file   = st.sidebar.file_uploader("ValidationData.csv (optional)", type=["csv"], key="val")

st.sidebar.markdown("---")
st.sidebar.header("Step 2 — Model Settings")
st.sidebar.caption("These control how the ML models are trained.")
n_trees   = st.sidebar.slider(
    "Random Forest: Number of Trees", 50, 300, 100, 50,
    help="More trees = more accurate but slower to train"
)
max_depth = st.sidebar.slider(
    "Random Forest: Max Tree Depth", 5, 30, 15, 5,
    help="How deep each decision tree can grow. Too deep = overfitting"
)
k_knn = st.sidebar.slider(
    "KNN: Number of Neighbours (k)", 1, 15, 5, 2,
    help="KNN looks at the k most similar past samples to make a prediction"
)
seed = st.sidebar.number_input(
    "Random Seed (keeps results consistent)", 0, 9999, 42, 1
)

st.sidebar.markdown("---")
st.sidebar.header("Step 3 — Visualization")
st.sidebar.caption("Filter the signal map and set the weak-signal threshold.")
floor_select = st.sidebar.selectbox(
    "Show Heatmap For", options=["All Floors", 0, 1, 2, 3, 4],
    help="Filter the signal heatmap to a single floor"
)
weak_thresh_input = st.sidebar.slider(
    "Weak Signal Threshold (dBm)", -90, -50, WEAK_THRESHOLD, 5,
    help="Any area with signal below this value is marked as a weak zone"
)
st.sidebar.caption(
    "Signal guide: above -60 dBm = good | below -70 dBm = poor | below -80 dBm = very poor"
)

st.sidebar.markdown("---")
st.sidebar.header("Step 4 — Live Monitor")
st.sidebar.caption("Simulates a device moving through the building in real time.")
rt_enabled     = st.sidebar.checkbox("Enable simulated real-time feed", value=False)
rt_interval    = st.sidebar.slider("Refresh every N seconds", 1, 10, 3, 1)
rt_noise_level = st.sidebar.slider(
    "Signal variation (dBm)", 1, 15, 5, 1,
    help="Adds random noise to simulate the signal naturally fluctuating"
)

# -------------------------------------------------------
# Landing screen before upload
# -------------------------------------------------------
if train_file is None:
    st.title("Wi-Fi Indoor Localization and Signal Analysis")
    st.caption("UJIIndoorLoc dataset  |  Random Forest + KNN  |  Coverage mapping  |  Simulated real-time monitoring")

    st.markdown("""
    <div class="explain-box">
    <b>What this project does:</b><br>
    Inside a building, GPS does not work. But your phone can detect Wi-Fi signals from
    routers around you. Different floors have different signal patterns — Floor 0 is close
    to some routers, Floor 3 has a completely different set of strong signals.<br><br>
    This system was trained on real Wi-Fi readings collected across a university building.
    It learns those patterns and can predict which floor a device is on, just from the
    signal strengths it picks up. It also maps out areas with weak coverage and suggests
    where new routers should be placed to fix them.
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "Upload **TrainingData.csv** in the sidebar to begin.\n\n"
        "Get the dataset from: https://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc"
    )
    st.stop()

# -------------------------------------------------------
# Data loading
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(file_bytes, label):
    df = pd.read_csv(io.BytesIO(file_bytes))
    wap_cols = [c for c in df.columns if c.startswith("WAP")]
    if not wap_cols:
        st.error(f"No WAP columns found in {label}.")
        st.stop()
    # 100 means "router not detected" in this dataset.
    # We replace it with -110 dBm so the model treats it as a very weak signal,
    # not a strong one (100 would look like a strong signal to the model).
    df[wap_cols] = df[wap_cols].replace(MISSING_RAW, MISSING_DBM)
    return df, wap_cols

with st.spinner("Loading and cleaning data..."):
    df_train, wap_cols = load_data(train_file.read(), "TrainingData")
    df_val = None
    if val_file is not None:
        df_val, _ = load_data(val_file.read(), "ValidationData")

for col in ["FLOOR", "LATITUDE", "LONGITUDE"]:
    if col not in df_train.columns:
        st.error(f"Column '{col}' not found. Is this the correct UJIIndoorLoc dataset?")
        st.stop()

# -------------------------------------------------------
# Feature preparation
# -------------------------------------------------------
def prepare_features(df_tr, df_v, all_wap_cols, var_threshold=VAR_FILTER):
    """
    Remove useless WAP columns and scale the remaining ones.
    A WAP column with almost no variation means that router was
    never detected anywhere — it gives the model nothing useful.
    """
    X_tr = df_tr[all_wap_cols].copy()
    stds = X_tr.std()
    keep = stds[stds >= var_threshold].index.tolist()
    if not keep:
        keep = all_wap_cols

    # StandardScaler: puts all WAP columns on the same scale
    # so no single router unfairly dominates the prediction
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr[keep])

    X_val_sc = None
    if df_v is not None:
        X_v = df_v[all_wap_cols].copy()
        for c in keep:
            if c not in X_v.columns:
                X_v[c] = MISSING_DBM
        X_val_sc = scaler.transform(X_v[keep])

    return X_tr_sc, X_val_sc, scaler, keep

X_train, X_val, scaler, kept_cols = prepare_features(df_train, df_val, wap_cols)
y_train = df_train["FLOOR"].values
y_val   = df_val["FLOOR"].values if df_val is not None else None

# -------------------------------------------------------
# Model training
# -------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_models(_X, y, n_trees, max_depth, k_knn, seed):
    rf = RandomForestClassifier(
        n_estimators=int(n_trees), max_depth=int(max_depth),
        random_state=int(seed), n_jobs=-1
    )
    rf.fit(_X, y)
    knn = KNeighborsClassifier(n_neighbors=int(k_knn), n_jobs=-1)
    knn.fit(_X, y)
    return rf, knn

with st.spinner("Training Random Forest and KNN models..."):
    rf_model, knn_model = train_models(X_train, y_train, n_trees, max_depth, k_knn, seed)

rf_train_preds  = rf_model.predict(X_train)
knn_train_preds = knn_model.predict(X_train)
rf_train_acc    = accuracy_score(y_train, rf_train_preds)
knn_train_acc   = accuracy_score(y_train, knn_train_preds)

rf_val_acc = knn_val_acc = None
rf_val_preds = knn_val_preds = None
if X_val is not None and y_val is not None:
    rf_val_preds  = rf_model.predict(X_val)
    knn_val_preds = knn_model.predict(X_val)
    rf_val_acc    = accuracy_score(y_val, rf_val_preds)
    knn_val_acc   = accuracy_score(y_val, knn_val_preds)

# -------------------------------------------------------
# Page title
# -------------------------------------------------------
st.title("Wi-Fi Indoor Localization and Signal Analysis")
st.caption(
    "Dataset: UJIIndoorLoc  |  Models: Random Forest + KNN  |  "
    "Task: Predict which floor a device is on from Wi-Fi signal readings"
)

# ================================================================
# SECTION 1: How well did the model learn?
# ================================================================
st.markdown('<div class="section-header">Section 1 — How Well Does the Model Predict the Floor?</div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="explain-box">
<b>What you are looking at:</b><br>
We trained two different ML models on the Wi-Fi signal data.
After training, we tested both models on samples they had never seen before (the validation set)
and measured how often they got the floor right.<br><br>
<b>Random Forest</b> works by building 100+ decision trees that each vote on the answer.
The majority vote wins. It handles noisy data well.<br>
<b>KNN (K-Nearest Neighbours)</b> works by finding the k most similar past readings
and copying their floor label. Simple but surprisingly effective here.<br><br>
<b>Accuracy</b> = percentage of test samples where the predicted floor matched the real floor.
A higher number is better. 95%+ is excellent for this dataset.
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Random Forest — Training Accuracy",  f"{rf_train_acc*100:.2f}%",
          help="How well RF did on data it was trained on")
c2.metric("KNN — Training Accuracy",            f"{knn_train_acc*100:.2f}%",
          help="How well KNN did on data it was trained on")
c3.metric("Random Forest — Validation Accuracy",
          f"{rf_val_acc*100:.2f}%"  if rf_val_acc  is not None else "Upload val file",
          help="How well RF did on NEW data it had never seen")
c4.metric("KNN — Validation Accuracy",
          f"{knn_val_acc*100:.2f}%" if knn_val_acc is not None else "Upload val file",
          help="How well KNN did on NEW data it had never seen")
c5.metric("WAP Features Used",
          f"{len(kept_cols)} of {len(wap_cols)}",
          help="Low-information WAP columns were removed before training")

if rf_val_preds is not None:
    st.markdown("""
    <div class="explain-box">
    <b>Confusion Matrix — how to read it:</b><br>
    Each row = what the real floor was. Each column = what the model predicted.
    The numbers on the diagonal (top-left to bottom-right) are correct predictions.
    Numbers off the diagonal are mistakes — e.g. the model said Floor 2 but it was really Floor 1.
    A good model has large numbers on the diagonal and near-zero everywhere else.
    </div>
    """, unsafe_allow_html=True)

    floor_labels = sorted(np.unique(y_val))
    label_names  = [f"Floor {f}" for f in floor_labels]

    fig_cm, axes_cm = plt.subplots(1, 2, figsize=(11, 4))
    fig_cm.patch.set_facecolor("#0e1117")

    for ax, preds, title in zip(
        axes_cm,
        [rf_val_preds, knn_val_preds],
        ["Random Forest", "KNN"]
    ):
        cm = confusion_matrix(y_val, preds, labels=floor_labels)
        ax.set_facecolor("#0e1117")
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(floor_labels)))
        ax.set_yticks(range(len(floor_labels)))
        ax.set_xticklabels(label_names, color="white", fontsize=8)
        ax.set_yticklabels(label_names, color="white", fontsize=8)
        ax.set_xlabel("What the model predicted", color="#aaaaaa")
        ax.set_ylabel("What the real floor was",  color="#aaaaaa")
        ax.set_title(f"Confusion Matrix — {title}", color="white", fontsize=10)
        ax.tick_params(colors="white")
        thresh = cm.max() / 2
        for i in range(len(floor_labels)):
            for j in range(len(floor_labels)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig_cm)

    with st.expander("Full Classification Report (precision, recall, F1 per floor)"):
        st.markdown("""
        <div class="explain-box">
        <b>Precision</b>: of all the times the model said "Floor X", how often was it right?<br>
        <b>Recall</b>: of all the actual Floor X samples, how many did the model catch?<br>
        <b>F1-score</b>: a single number combining both. Closer to 1.0 is better.<br>
        <b>Support</b>: how many real samples exist for each floor in the validation set.
        </div>
        """, unsafe_allow_html=True)
        st.code(classification_report(
            y_val, rf_val_preds,
            target_names=[f"Floor {f}" for f in floor_labels]
        ))

# -------------------------------------------------------
# Helpers: grid computation, router suggestion, plot
# -------------------------------------------------------
def compute_signal_grid(df, floor_filter="All Floors", bins=GRID_BINS):
    """
    Takes all measurement samples, bins them into a lat/lon grid,
    and computes the average best signal strength per grid cell.
    """
    data = df.copy()
    if floor_filter != "All Floors":
        data = data[data["FLOOR"] == int(floor_filter)]
    if len(data) == 0:
        return None, None, None, None

    wap_data    = data[[c for c in data.columns if c.startswith("WAP")]].values
    best_signal = np.where(wap_data == MISSING_DBM, np.nan, wap_data)
    best_signal = np.nanmax(best_signal, axis=1)  # strongest router at each point

    lats = data["LATITUDE"].values
    lons = data["LONGITUDE"].values

    lat_edges = np.linspace(lats.min(), lats.max(), bins + 1)
    lon_edges = np.linspace(lons.min(), lons.max(), bins + 1)

    grid   = np.full((bins, bins), np.nan)
    counts = np.zeros((bins, bins))

    for idx in range(len(lats)):
        if np.isnan(best_signal[idx]):
            continue
        lat_i = min(np.searchsorted(lat_edges[1:], lats[idx]), bins - 1)
        lon_i = min(np.searchsorted(lon_edges[1:], lons[idx]), bins - 1)
        if np.isnan(grid[lat_i, lon_i]):
            grid[lat_i, lon_i] = 0.0
        grid[lat_i, lon_i] += best_signal[idx]
        counts[lat_i, lon_i] += 1

    with np.errstate(invalid="ignore"):
        grid = np.where(counts > 0, grid / counts, np.nan)

    # Gaussian smoothing makes the heatmap look continuous instead of blocky
    valid_mask = ~np.isnan(grid)
    fill_val   = np.nanmean(grid) if not np.all(np.isnan(grid)) else 0.0
    temp       = np.where(np.isnan(grid), fill_val, grid)
    smoothed   = gaussian_filter(temp, sigma=SMOOTH_SIGMA)
    grid_final = np.where(valid_mask, smoothed, np.nan)
    weak_mask  = grid_final < weak_thresh_input

    return grid_final, lat_edges, lon_edges, weak_mask


def suggest_router_positions(weak_mask, lat_centers, lon_centers, top_n=3):
    """
    Finds the largest clusters of weak-signal cells using BFS
    (the same flood-fill idea as MS Paint's bucket tool).
    Returns the center of each cluster as a suggested router location.
    """
    weak_coords = np.argwhere(weak_mask)
    if len(weak_coords) == 0:
        return []

    visited  = np.zeros_like(weak_mask, dtype=bool)
    clusters = []

    for start in weak_coords:
        r, c = int(start[0]), int(start[1])
        if visited[r, c]:
            continue
        cluster = []
        queue   = deque([(r, c)])
        visited[r, c] = True
        while queue:
            cr, cc = queue.popleft()
            cluster.append((cr, cc))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nr, nc = cr + dr, cc + dc
                if (0 <= nr < weak_mask.shape[0] and
                    0 <= nc < weak_mask.shape[1] and
                    weak_mask[nr, nc] and not visited[nr, nc]):
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        clusters.append(cluster)

    clusters.sort(key=len, reverse=True)

    suggestions = []
    for cluster in clusters[:top_n]:
        rows  = [p[0] for p in cluster]
        cols  = [p[1] for p in cluster]
        lat_s = float(np.mean([lat_centers[r] for r in rows]))
        lon_s = float(np.mean([lon_centers[c] for c in cols]))
        suggestions.append((lat_s, lon_s, len(cluster)))
    return suggestions


def draw_heatmap_figure(grid, lat_edges, lon_edges, weak_mask, floor_label):
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    LON, LAT    = np.meshgrid(lon_centers, lat_centers)

    total_valid = int(np.sum(~np.isnan(grid)))
    n_weak      = int(np.sum(weak_mask & ~np.isnan(grid)))
    weak_pct    = 100.0 * n_weak / total_valid if total_valid > 0 else 0.0

    suggestions = suggest_router_positions(weak_mask, lat_centers, lon_centers, top_n=3)

    fig = plt.figure(figsize=(13, 5.5), facecolor="#0e1117")
    gs  = GridSpec(1, 2, figure=fig, wspace=0.35)

    # Left panel: colour-coded signal strength across the floor plan
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#0e1117")

    vmin    = float(np.nanpercentile(grid, 5))
    vmax    = float(np.nanpercentile(grid, 95))
    vcenter = float(np.clip(weak_thresh_input, vmin + 0.01, vmax - 0.01))
    norm    = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    cf = ax1.contourf(LON, LAT, grid, levels=20, cmap="RdYlGn", norm=norm)
    ax1.contour(LON, LAT, grid, levels=[weak_thresh_input],
                colors="#4a9eff", linewidths=1.2, linestyles="--")

    cbar = plt.colorbar(cf, ax=ax1, fraction=0.04, pad=0.04)
    cbar.set_label("Average Best Signal (dBm)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    for i, (lat_s, lon_s, _) in enumerate(suggestions):
        ax1.scatter(lon_s, lat_s, s=120, c="#00cfff", marker="*",
                    edgecolors="white", linewidths=0.6, zorder=6)
        ax1.annotate(f"Router {i+1}", (lon_s, lat_s),
                     textcoords="offset points", xytext=(5, 4),
                     color="#00cfff", fontsize=7, fontweight="bold")

    ax1.set_title(f"Signal Strength Map — {floor_label}", color="white", fontsize=10)
    ax1.set_xlabel("Longitude (building east-west position)", color="#aaaaaa", fontsize=7.5)
    ax1.set_ylabel("Latitude (building north-south position)", color="#aaaaaa", fontsize=7.5)
    ax1.tick_params(colors="white", labelsize=7)
    for sp in ax1.spines.values():
        sp.set_edgecolor("#333344")
    ax1.legend(handles=[
        mpatches.Patch(facecolor="#2ecc40", label="Strong signal (good coverage)"),
        mpatches.Patch(facecolor="#e74c3c", label="Weak signal (poor coverage)"),
        plt.Line2D([0],[0], color="#4a9eff", lw=1.2, ls="--",
                   label=f"Weak zone boundary ({weak_thresh_input} dBm)"),
        plt.Line2D([0],[0], marker="*", color="w", markerfacecolor="#00cfff",
                   markersize=9, label="Suggested new router location", linestyle="None"),
    ], fontsize=7, loc="lower right", facecolor="#1a1a2e",
       labelcolor="white", edgecolor="#333355")

    # Right panel: binary weak/strong map — easier to read at a glance
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#0e1117")

    weak_display = np.where(np.isnan(grid), np.nan, np.where(weak_mask, 1.0, 0.0))
    cmap_weak    = mcolors.ListedColormap(["#1a3a1a", "#cc2222"])
    ax2.contourf(LON, LAT, weak_display, levels=[-0.5, 0.5, 1.5], cmap=cmap_weak)

    for i, (lat_s, lon_s, _) in enumerate(suggestions):
        ax2.scatter(lon_s, lat_s, s=140, c="#00cfff", marker="*",
                    edgecolors="white", linewidths=0.6, zorder=6)
        ax2.annotate(f"Router {i+1}", (lon_s, lat_s),
                     textcoords="offset points", xytext=(5, 4),
                     color="#00cfff", fontsize=7, fontweight="bold")

    ax2.set_title("Weak Zone Map — Red = needs a router", color="white", fontsize=10)
    ax2.set_xlabel("Longitude", color="#aaaaaa", fontsize=7.5)
    ax2.set_ylabel("Latitude",  color="#aaaaaa", fontsize=7.5)
    ax2.tick_params(colors="white", labelsize=7)
    for sp in ax2.spines.values():
        sp.set_edgecolor("#333344")
    ax2.legend(handles=[
        mpatches.Patch(facecolor="#cc2222", label="Weak zone (signal below threshold)"),
        mpatches.Patch(facecolor="#1a3a1a", label="Good coverage (signal above threshold)"),
        plt.Line2D([0],[0], marker="*", color="w", markerfacecolor="#00cfff",
                   markersize=9, label="Suggested router location", linestyle="None"),
    ], fontsize=7, loc="lower right", facecolor="#1a1a2e",
       labelcolor="white", edgecolor="#333355")

    plt.tight_layout()
    return fig, suggestions, weak_pct, n_weak


# ================================================================
# SECTION 2: Signal coverage heatmap
# ================================================================
st.markdown('<div class="section-header">Section 2 — Where is the Wi-Fi Signal Strong or Weak?</div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="explain-box">
<b>What you are looking at:</b><br>
This is a top-down map of the building floor plan. Each colour shows the average Wi-Fi signal
strength at that physical location, based on all the measurements in the dataset.<br><br>
<b>Green</b> = strong signal. The device connects well here.<br>
<b>Red</b> = weak signal. The device would struggle to connect or might drop the connection.<br>
<b>Blue dashed line</b> = the threshold boundary. Everything to the red side is a "weak zone."<br>
<b>Cyan stars (R1, R2, R3)</b> = suggested locations for new routers, placed at the center of
the largest weak areas. Adding a router there would fix the most coverage at once.<br><br>
The right map simplifies this to a pure red/green binary — red areas need a router.
</div>
""", unsafe_allow_html=True)

grid, lat_edges, lon_edges, weak_mask = compute_signal_grid(df_train, floor_select, GRID_BINS)

if grid is None:
    st.warning("No data available for the selected floor. Try 'All Floors' or a different floor number.")
else:
    floor_label = f"Floor {floor_select}" if floor_select != "All Floors" else "All Floors"
    fig_map, suggestions, weak_pct, n_weak = draw_heatmap_figure(
        grid, lat_edges, lon_edges, weak_mask, floor_label
    )
    st.pyplot(fig_map)

    # ================================================================
    # SECTION 3: Coverage numbers
    # ================================================================
    st.markdown('<div class="section-header">Section 3 — Coverage Numbers at a Glance</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-box">
    These four numbers summarise the signal quality across the entire mapped area.
    <b>Weak Zone Coverage</b> is the most important one — it tells you what percentage of
    the floor has poor Wi-Fi. A number above 20% suggests the network needs attention.
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Weak Zone Coverage",
              f"{weak_pct:.1f}%",
              help="Percentage of the mapped area with signal below the threshold")
    s2.metric("Average Signal Strength",
              f"{np.nanmean(grid):.1f} dBm",
              help="Mean signal across all measured locations. Closer to 0 is stronger.")
    s3.metric("Strongest Signal Found",
              f"{np.nanmax(grid):.1f} dBm",
              help="The best signal anywhere on this floor")
    s4.metric("Weakest Signal Found",
              f"{np.nanmin(grid):.1f} dBm",
              help="The worst signal anywhere on this floor")

    # ================================================================
    # SECTION 4: Router placement suggestions
    # ================================================================
    st.markdown('<div class="section-header">Section 4 — Where Should New Routers Go?</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-box">
    <b>How this works:</b><br>
    The algorithm scans the map for red (weak) cells and groups neighbouring red cells
    together into clusters — the same way a bucket fill tool groups connected pixels.
    It then finds the geographic center of the top 3 largest clusters.<br><br>
    Placing a router at that center point would cover the most weak area at once.
    This is a <b>greedy coverage heuristic</b> — not mathematically perfect, but fast,
    explainable, and practical.
    </div>
    """, unsafe_allow_html=True)

    if suggestions:
        sug_df = pd.DataFrame([
            {
                "Priority":                   f"Router {i+1}",
                "Suggested Latitude":         round(lat_s, 4),
                "Suggested Longitude":        round(lon_s, 4),
                "Weak Grid Cells Covered":    sz,
                "Estimated Coverage Gain":    f"{100*sz/n_weak:.1f}% of total weak area"
                                              if n_weak > 0 else "N/A"
            }
            for i, (lat_s, lon_s, sz) in enumerate(suggestions)
        ])
        st.dataframe(sug_df, use_container_width=True, hide_index=True)
        st.caption(
            "Router 1 has the highest priority — it covers the single largest cluster of "
            "weak cells. Routers 2 and 3 address the next two largest clusters."
        )
    else:
        st.success("No significant weak zones detected at the current threshold. Coverage looks good.")

# ================================================================
# SECTION 5: Per-floor signal distribution
# ================================================================
st.markdown('<div class="section-header">Section 5 — Signal Distribution per Floor</div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="explain-box">
<b>What you are looking at:</b><br>
Each chart shows, for one floor, how the signal strength is distributed across all
measurement points on that floor. Think of it like a histogram of scores.<br><br>
<b>Tall bars on the right</b> = most places on that floor have good signal.<br>
<b>Tall bars on the left</b> = most places have weak signal — that floor needs attention.<br>
The <b>red dashed line</b> is the weak-signal threshold. Bars to the left of it are weak zones.
The <b>green dotted line</b> is the average signal on that floor.
The percentage in the title shows what fraction of that floor is below the threshold.
</div>
""", unsafe_allow_html=True)

floors_available = sorted(df_train["FLOOR"].unique())
n_floors         = len(floors_available)

fig_floors, floor_axes = plt.subplots(
    1, n_floors, figsize=(4 * n_floors, 3.8), facecolor="#0e1117"
)
if n_floors == 1:
    floor_axes = [floor_axes]

for ax, floor in zip(floor_axes, floors_available):
    ax.set_facecolor("#0e1117")
    floor_df  = df_train[df_train["FLOOR"] == floor]
    wap_data  = floor_df[[c for c in floor_df.columns if c.startswith("WAP")]].values
    best      = np.where(wap_data == MISSING_DBM, np.nan, wap_data)
    best_vals = np.nanmax(best, axis=1)
    best_vals = best_vals[~np.isnan(best_vals)]

    ax.hist(best_vals, bins=30, color="#4a9eff", edgecolor="#0e1117", alpha=0.85)
    ax.axvline(weak_thresh_input, color="#ff4466", lw=1.2, ls="--",
               label=f"Weak threshold ({weak_thresh_input} dBm)")
    ax.axvline(float(np.mean(best_vals)), color="#44ff88", lw=1.0, ls=":",
               label=f"Floor average: {np.mean(best_vals):.1f} dBm")

    weak_frac = 100.0 * float(np.sum(best_vals < weak_thresh_input)) / len(best_vals)
    ax.set_title(f"Floor {floor}  —  {weak_frac:.0f}% weak", color="white", fontsize=9)
    ax.set_xlabel("Best signal strength (dBm)", color="#aaaaaa", fontsize=7.5)
    ax.set_ylabel("Number of measurement points", color="#aaaaaa", fontsize=7.5)
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333344")
    ax.legend(fontsize=6.5, facecolor="#1a1a2e", labelcolor="white", edgecolor="#333355")

plt.tight_layout()
st.pyplot(fig_floors)

# ================================================================
# SECTION 6: Feature importance
# ================================================================
with st.expander("Section 6 — Which Routers Matter Most for Floor Prediction?"):
    st.markdown("""
    <div class="explain-box">
    The Random Forest model can tell us which WAP access points it relied on most
    when making its floor predictions. A high importance score means that router's
    signal was a strong clue about which floor the device was on.<br><br>
    Access points near stairwells or elevators often rank high because their signal
    changes significantly between floors. Access points in the middle of a floor may
    rank low because they are equally detectable from multiple floors.
    </div>
    """, unsafe_allow_html=True)

    imp       = rf_model.feature_importances_
    top_idx   = np.argsort(imp)[::-1][:25]
    top_names = [kept_cols[i] for i in top_idx]
    top_vals  = imp[top_idx]

    fig_imp, ax_imp = plt.subplots(figsize=(11, 3.5), facecolor="#0e1117")
    ax_imp.set_facecolor("#0e1117")
    ax_imp.bar(range(25), top_vals, color="#4a9eff", edgecolor="#0e1117", width=0.75)
    ax_imp.set_xticks(range(25))
    ax_imp.set_xticklabels(top_names, rotation=45, ha="right", fontsize=7.5, color="white")
    ax_imp.set_ylabel("Importance score\n(higher = more useful for prediction)",
                      color="#aaaaaa", fontsize=8)
    ax_imp.set_title("Top 25 Most Important Wi-Fi Access Points (WAPs) — Random Forest",
                     color="white", fontsize=10)
    ax_imp.tick_params(axis="y", colors="white")
    for sp in ax_imp.spines.values():
        sp.set_edgecolor("#333344")
    plt.tight_layout()
    st.pyplot(fig_imp)

# ================================================================
# SECTION 7: Dataset summary
# ================================================================
with st.expander("Section 7 — Dataset Summary and Raw Data Preview"):
    st.markdown("""
    <div class="explain-box">
    <b>Training set</b>: the samples the model learned from.<br>
    <b>Validation set</b>: samples the model had never seen — used to test real-world accuracy.<br>
    <b>Raw WAP columns</b>: 520 access points in the original data.<br>
    <b>After variance filter</b>: columns that were almost always -110 (never detected) were removed.
    </div>
    """, unsafe_allow_html=True)

    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Training Samples",      f"{len(df_train):,}")
    d2.metric("Validation Samples",    f"{len(df_val):,}" if df_val is not None else "N/A")
    d3.metric("Raw WAP Columns",       len(wap_cols))
    d4.metric("WAPs After Filtering",  len(kept_cols))
    d5.metric("Floor Classes",         len(floors_available))

    st.caption("First 8 rows of training data:")
    st.dataframe(df_train.head(8), use_container_width=True)

# ================================================================
# SECTION 8: Simulated real-time monitor
# ================================================================
st.markdown('<div class="section-header">Section 8 — Simulated Real-Time Floor Detection</div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="explain-box">
<b>What this is:</b><br>
This simulates what the system would look like if it were running live — for example,
tracking a person's phone as they move through the building.<br><br>
Every few seconds it picks a random Wi-Fi reading from the dataset, adds a small amount
of random noise (to simulate the signal naturally fluctuating), and runs it through the
trained model to predict which floor the device is on.<br><br>
The chart updates live, showing predicted floor vs actual floor over time, and whether
the signal was above or below the weak threshold.<br><br>
<b>Note:</b> This is simulated — no real hardware is involved. The data comes from the
dataset, not a live router. This is standard practice for demonstrating how such a system
would behave in deployment.
</div>
""", unsafe_allow_html=True)

if not rt_enabled:
    st.info(
        "Tick the **Enable simulated real-time feed** checkbox in the sidebar to start. "
        "The dashboard will auto-refresh every few seconds showing live predictions."
    )
else:
    rt_source   = df_val if df_val is not None else df_train
    rt_wap_vals = rt_source[[c for c in rt_source.columns if c.startswith("WAP")]].values
    rt_floors   = rt_source["FLOOR"].values

    if "rt_history" not in st.session_state:
        st.session_state.rt_history = {
            "tick": [], "true_floor": [], "pred_floor": [],
            "confidence": [], "avg_signal": [],
        }

    history = st.session_state.rt_history

    rng        = np.random.default_rng()
    idx        = int(rng.integers(0, len(rt_wap_vals)))
    raw_sample = rt_wap_vals[idx].copy().astype(float)
    detected   = raw_sample != MISSING_DBM

    if detected.any():
        raw_sample[detected] += rng.normal(0, rt_noise_level, size=int(detected.sum()))
    raw_sample = np.clip(raw_sample, -110, -10)

    true_floor = int(rt_floors[idx])
    avg_sig    = float(np.mean(raw_sample[detected])) if detected.any() else float(MISSING_DBM)

    sample_series = pd.Series(raw_sample, index=wap_cols)
    sample_df     = pd.DataFrame([sample_series[kept_cols]])
    sample_scaled = scaler.transform(sample_df)

    pred_floor = int(rf_model.predict(sample_scaled)[0])
    confidence = float(np.max(rf_model.predict_proba(sample_scaled)[0])) * 100

    tick_num = len(history["tick"]) + 1
    history["tick"].append(tick_num)
    history["true_floor"].append(true_floor)
    history["pred_floor"].append(pred_floor)
    history["confidence"].append(confidence)
    history["avg_signal"].append(avg_sig)
    for key in history:
        history[key] = history[key][-30:]

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Model Says: Floor", str(pred_floor))
    r2.metric(
        "Actually Floor", str(true_floor),
        delta="Correct prediction" if pred_floor == true_floor else "Wrong prediction",
        delta_color="normal" if pred_floor == true_floor else "inverse"
    )
    r3.metric("Model Confidence", f"{confidence:.1f}%",
              help="How sure the model is — 100% = completely certain")
    r4.metric("Sample Avg Signal", f"{avg_sig:.1f} dBm",
              help="Average signal of the detected routers in this reading")

    hist_df = pd.DataFrame(history)

    fig_rt, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 4.5), facecolor="#0e1117", sharex=True
    )
    fig_rt.subplots_adjust(hspace=0.08)

    for ax in (ax_top, ax_bot):
        ax.set_facecolor("#111827")
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a3a")
        ax.tick_params(colors="#aaaaaa", labelsize=7)

    ax_top.step(hist_df["tick"], hist_df["true_floor"],
                color="#44ff88", lw=1.2, label="Actual floor (ground truth)", where="post")
    ax_top.step(hist_df["tick"], hist_df["pred_floor"],
                color="#4a9eff", lw=1.2, ls="--", label="Predicted floor (model output)", where="post")
    ax_top.set_ylabel("Floor Number", color="#aaaaaa", fontsize=8)
    ax_top.set_title("Live Floor Prediction — Green = actual, Blue dashed = predicted",
                     color="white", fontsize=10)
    ax_top.legend(fontsize=7.5, facecolor="#1a1a2e", labelcolor="white",
                  edgecolor="#333355", loc="upper left")
    ax_top.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    avg_arr = np.array(hist_df["avg_signal"])
    ax_bot.plot(hist_df["tick"], avg_arr, color="#f0a500", lw=1.2,
                label="Average detected signal strength")
    ax_bot.axhline(weak_thresh_input, color="#ff4466", lw=0.9, ls="--",
                   label=f"Weak signal threshold ({weak_thresh_input} dBm)")
    ax_bot.fill_between(hist_df["tick"], avg_arr, weak_thresh_input,
                        where=avg_arr < weak_thresh_input,
                        alpha=0.25, color="#cc2222", label="Signal below threshold (red zone)")
    ax_bot.set_ylabel("Signal Strength (dBm)", color="#aaaaaa", fontsize=8)
    ax_bot.set_xlabel("Reading number (tick)", color="#aaaaaa", fontsize=8)
    ax_bot.legend(fontsize=7.5, facecolor="#1a1a2e", labelcolor="white",
                  edgecolor="#333355", loc="upper left")

    plt.tight_layout()
    st.pyplot(fig_rt)

    with st.expander("Last 10 individual predictions"):
        n = min(10, len(history["tick"]))
        log_df = pd.DataFrame({
            "Reading #":   history["tick"][-n:],
            "Actual Floor":  history["true_floor"][-n:],
            "Predicted Floor": history["pred_floor"][-n:],
            "Confidence":  [f"{v:.1f}%" for v in history["confidence"][-n:]],
            "Avg Signal":  [f"{v:.1f} dBm" for v in history["avg_signal"][-n:]],
            "Result":      [
                "Correct" if t == p else "Wrong"
                for t, p in zip(history["true_floor"][-n:], history["pred_floor"][-n:])
            ]
        })
        st.dataframe(log_df, use_container_width=True, hide_index=True)

    st.caption(
        "Simulated feed — replaying dataset samples with added signal noise. "
        "No live hardware involved."
    )

    time.sleep(int(rt_interval))
    st.rerun()

# ================================================================
# Project overview (collapsible)
# ================================================================
with st.expander("Project Overview — What This System Does and How"):
    st.markdown("""
    ### The Problem
    GPS does not work inside buildings. But Wi-Fi signals do. Different locations in a
    building produce different patterns of signal strengths from nearby routers.
    If you collect enough labeled examples — "at this GPS location on Floor 2, here is the
    signal from each of the 520 access points" — a machine learning model can learn
    those patterns and predict which floor any new reading came from.

    ### The Dataset
    UJIIndoorLoc (UCI Machine Learning Repository, 2014). Collected across 3 university
    buildings with 4–5 floors each. Each row is one measurement: 520 WAP signal values
    plus the known floor, latitude, and longitude.

    ### Data Cleaning
    The dataset uses 100 as a placeholder for "router not detected." This is replaced with
    -110 dBm — a realistic very-weak-signal value — so the model interprets it correctly.
    WAP columns with near-zero variance are removed since they carry no useful information.

    ### The Models
    **Random Forest**: builds many decision trees on random subsets of data, then takes
    a majority vote. Robust to noisy data. Provides feature importance scores.

    **KNN**: stores all training samples. For a new reading, finds the k most similar
    past samples and votes on their floor label. Simple and effective on fingerprint data.

    ### Coverage Mapping
    Measurements are plotted onto a lat/lon grid. Average signal per grid cell is computed
    and Gaussian-smoothed. Cells below the threshold are flagged as weak zones. BFS
    flood-fill groups weak cells into clusters, and each cluster center becomes a
    suggested router placement location.

    ### Simulated Real-Time Monitor
    Replays dataset rows at a timed interval with added Gaussian noise to simulate signal
    fluctuation. Demonstrates how the system would behave with live sensor data.
    Clearly labelled as simulated throughout.
    """)