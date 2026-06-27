#!/usr/bin/env python3
"""
train.py - Offline Training Pipeline
Spotify Popularity Predictor
Author: Amila Herenda

Run once on your machine:
    python train.py

Outputs:
    eda_plots/*.png         - EDA visualisations (saved as PNG, not displayed)
    models/classifier.pkl  - full scikit-learn Pipeline (preprocessor + model)
    models/regressor.pkl   - full scikit-learn Pipeline (preprocessor + model)

The app (api.py) loads these .pkl files at startup and never retrains.
"""

# =============================================================================
# 0.  IMPORTS & CONFIGURATION
# =============================================================================
import os
import warnings
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # non-interactive backend - works on servers without a display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    r2_score, mean_absolute_error, mean_squared_error,
)
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")   # suppress sklearn convergence / XGBoost version warnings

# ── Global constants - change ONE place, affects the whole project ─────────────
SEED                 = 42    # random_state used everywhere -> reproducible results
POPULARITY_THRESHOLD = 70    # popularity >= 70  ->  label 1 (popular)
TEST_SIZE            = 0.20  # 20 % of rows held out for final evaluation
N_SELECT             = 15    # top-N features kept by SelectKBest for linear models

DATA_PATH  = os.path.join("data",      "dataset.csv")
MODELS_DIR = "models"
PLOTS_DIR  = "eda_plots"

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 100


# =============================================================================
# 1.  LOAD DATA
# =============================================================================
def load_data() -> pd.DataFrame:
    """
    Read dataset.csv and print a quick overview.
    No transformations here - we want to see the raw data first.
    """
    print("\n" + "=" * 60)
    print("STEP 1 - LOAD DATA")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"\n[ERROR] {DATA_PATH} not found.\n"
            "Download dataset.csv from Kaggle and place it in the data/ folder.\n"
            "URL: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset"
        )

    df = pd.read_csv(DATA_PATH)
    print(f"Shape        : {df.shape[0]:,} rows  x  {df.shape[1]} columns")
    print(f"Memory usage : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print("\nColumn dtypes:")
    print(df.dtypes.value_counts().to_string())
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
    return df


# =============================================================================
# 2.  EXPLORATORY DATA ANALYSIS
# =============================================================================
def _save_fig(filename: str) -> None:
    """Save the current matplotlib figure to PLOTS_DIR and close it."""
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {path}")


def run_eda(df: pd.DataFrame) -> None:
    """
    Generate and save EDA plots.

    WHY run EDA before cleaning?
    We want to see the data exactly as it came from Kaggle - warts and all.
    If we cleaned first, we would lose visibility into how bad the raw data was.
    Professor's principle: 'bad data analysis = model learns the wrong thing.'
    """
    print("\n" + "=" * 60)
    print("STEP 2 - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # ── Missing values & duplicates ───────────────────────────────────────────
    missing = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing[missing > 0].to_string() if missing.any() else "  None found.")

    dupes = df.duplicated().sum()
    print(f"\nDuplicate rows: {dupes:,}")

    # ── Plot 1: Target variable - popularity distribution ─────────────────────
    print("\nPlot 1: Popularity distribution ...")
    binary = (df["popularity"] >= POPULARITY_THRESHOLD).astype(int)
    counts = binary.value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df["popularity"], bins=60, color="steelblue", edgecolor="white")
    axes[0].axvline(POPULARITY_THRESHOLD, color="crimson", linestyle="--",
                    label=f"Threshold = {POPULARITY_THRESHOLD}")
    axes[0].set_xlabel("Popularity score (0-100)")
    axes[0].set_ylabel("Track count")
    axes[0].set_title("Popularity Distribution (raw)")
    axes[0].legend()

    bar_labels = ["Not popular (0)", "Popular (1)"]
    bar_colors = ["#d9534f", "#5cb85c"]
    axes[1].bar(bar_labels, counts.values, color=bar_colors)
    axes[1].set_ylabel("Track count")
    axes[1].set_title(f"Class Balance  (threshold >= {POPULARITY_THRESHOLD})")
    for i, v in enumerate(counts.values):
        axes[1].text(i, v + 300, f"{v:,}\n({v / len(df) * 100:.1f} %)",
                     ha="center", fontsize=9)

    plt.suptitle("Target Variable - Popularity", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_fig("01_popularity_distribution.png")

    # ── Plot 2: Numeric feature distributions ─────────────────────────────────
    print("Plot 2: Feature distributions ...")
    num_cols = [c for c in df.select_dtypes(include=np.number).columns
                if c != "popularity"]
    n_cols, n_rows = 4, int(np.ceil(len(num_cols) / 4))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        axes[i].hist(df[col].dropna(), bins=40, color="steelblue", edgecolor="white")
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(labelsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Numeric Feature Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_fig("02_feature_distributions.png")

    # ── Plot 3: Correlation heatmap ───────────────────────────────────────────
    print("Plot 3: Correlation heatmap ...")
    corr_cols = num_cols + ["popularity"]
    corr = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))   # show lower triangle only

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
        annot_kws={"size": 7},
    )
    ax.set_title("Pearson Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_fig("03_correlation_heatmap.png")

    # ── Plot 4: Outlier boxplots ──────────────────────────────────────────────
    print("Plot 4: Outlier boxplots ...")
    box_cols = ["danceability", "energy", "loudness", "speechiness",
                "acousticness", "instrumentalness", "liveness",
                "valence", "tempo", "duration_ms"]
    box_cols = [c for c in box_cols if c in df.columns]

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()
    for i, col in enumerate(box_cols):
        axes[i].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                        boxprops=dict(facecolor="steelblue", alpha=0.6),
                        medianprops=dict(color="crimson"))
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(labelsize=7)
    for j in range(len(box_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Boxplots - Outlier Detection", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_fig("04_outlier_boxplots.png")

    # ── Plot 5: Feature-target scatter ───────────────────────────────────────
    print("Plot 5: Feature-target relationships ...")
    scatter_cols = ["danceability", "energy", "loudness", "valence",
                    "acousticness", "speechiness", "tempo", "instrumentalness"]
    scatter_cols = [c for c in scatter_cols if c in df.columns]

    # sample 5 000 rows so the plot renders quickly and dots are visible
    sample = df.sample(min(5_000, len(df)), random_state=SEED)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(scatter_cols):
        axes[i].scatter(sample[col], sample["popularity"],
                        alpha=0.15, s=5, color="steelblue")
        axes[i].set_xlabel(col, fontsize=8)
        axes[i].set_ylabel("popularity", fontsize=8)
        axes[i].tick_params(labelsize=7)
    plt.suptitle("Audio Feature vs Popularity Score", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_fig("05_feature_target_scatter.png")

    # ── Plot 6: Top genres by average popularity ──────────────────────────────
    if "track_genre" in df.columns:
        print("Plot 6: Genre popularity ...")
        genre_pop = (
            df.groupby("track_genre")["popularity"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
            .head(25)
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(genre_pop.index[::-1], genre_pop["mean"][::-1], color="steelblue")
        ax.set_xlabel("Mean Popularity Score")
        ax.set_title("Top 25 Genres by Average Popularity", fontsize=13, fontweight="bold")
        plt.tight_layout()
        _save_fig("06_genre_popularity.png")

    print("\nEDA complete - all plots saved to eda_plots/")


# =============================================================================
# 3.  DATA CLEANING
# =============================================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame and return the cleaned version.

    Decisions (be ready to explain each one to your professor):

    Drop identifier columns:
        track_id, artists, album_name, track_name are names/IDs, not audio
        features. A model trained on artist names would only work for artists
        already in the dataset - it wouldn't generalise to new artists. That
        is a form of data leakage through the identity of the song, not its
        sound. We want the model to predict popularity from AUDIO only.

    Drop duplicates:
        The same row repeated twice adds no new information but inflates
        metrics - the model appears better than it is because test rows
        may have appeared in training.

    Missing values:
        We check and drop rows with NaN. If the dataset has very few missing
        values (< 0.1 %), dropping is safe and simple. If it had many, we
        would use imputation instead.

    Convert explicit bool -> int:
        scikit-learn cannot handle Python booleans in ColumnTransformer
        numeric branches. Casting to int (0/1) makes it numeric.
    """
    print("\n" + "=" * 60)
    print("STEP 3 - DATA CLEANING")
    print("=" * 60)

    # ── Drop identifier / free-text columns ───────────────────────────────────
    drop_cols = ["Unnamed: 0", "track_id", "artists", "album_name", "track_name"]
    existing = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing)
    print(f"Dropped identifier columns: {existing}")

    # ── Drop duplicate rows ───────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df):,} duplicate rows  ->  {len(df):,} remaining")

    # ── Missing values ────────────────────────────────────────────────────────
    missing_total = df.isnull().sum().sum()
    if missing_total > 0:
        print(f"Missing values found: {missing_total}  ->  dropping affected rows")
        df = df.dropna()
        print(f"After dropna: {len(df):,} rows")
    else:
        print("No missing values found - nothing to drop.")

    # ── Convert bool -> int ───────────────────────────────────────────────────
    if "explicit" in df.columns and df["explicit"].dtype == bool:
        df["explicit"] = df["explicit"].astype(int)
        print("Converted `explicit`  bool -> int  (0 = clean, 1 = explicit)")

    print(f"\nClean data shape: {df.shape}")
    return df.reset_index(drop=True)


# =============================================================================
# 4.  FEATURE ENGINEERING
# =============================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features derived from existing columns.

    WHY engineer features at all?
    Raw audio features measure individual dimensions. Their combinations can
    capture relationships the model might otherwise miss. For example,
    `energy * danceability` is high only when a song is both energetic AND
    danceable - neither column alone captures that joint property.

    IMPORTANT: These are pure arithmetic transformations of existing columns.
    No external data, no statistics from other rows. Therefore there is
    NO data leakage risk from engineering at this stage.

    What we do NOT do here: scaling, encoding, or any step that requires
    computing a statistic across rows (mean, std, etc.). Those steps go
    inside the Pipeline so they are fitted on training data only.
    """
    print("\n" + "=" * 60)
    print("STEP 4 - FEATURE ENGINEERING")
    print("=" * 60)

    df = df.copy()

    # How energetic is the track relative to how acoustic it is?
    # High value -> electronic / driving; low value -> mellow / unplugged
    df["energy_acoustic_ratio"] = df["energy"] / (df["acousticness"] + 1e-6)

    # Tracks that are BOTH energetic AND danceable tend to be commercial hits
    df["dance_energy"] = df["danceability"] * df["energy"]

    # Tracks that are BOTH happy (high valence) AND energetic -> upbeat
    df["valence_energy"] = df["valence"] * df["energy"]

    # Effective beat density: tempo (BPM) adjusted for time signature
    # A song at 120 BPM in 4/4 has the same feel as 90 BPM in 3/4
    df["tempo_beat_ratio"] = df["tempo"] / df["time_signature"].clip(lower=1)

    # Duration in minutes - milliseconds are hard to interpret
    df["duration_min"] = df["duration_ms"] / 60_000

    new_cols = [
        "energy_acoustic_ratio", "dance_energy", "valence_energy",
        "tempo_beat_ratio", "duration_min",
    ]
    print(f"Created {len(new_cols)} engineered features: {new_cols}")
    print(f"Data shape after engineering: {df.shape}")
    return df


# =============================================================================
# 5.  CREATE TARGETS & STRATIFIED SPLIT
# =============================================================================
def prepare_and_split(df: pd.DataFrame):
    """
    Create the two target vectors and split into train / test.

    WHY 80 / 20?
    A larger training set gives the models more data to learn from.
    20 % is enough rows for reliable metric estimates on the test set
    (roughly 22 000 rows -> stable F1, R2, etc.).

    WHY stratify on y_clf?
    Only ~10-15 % of songs have popularity >= 70. Without stratification,
    random splitting might give you a test set with 8 % positives and a
    train set with 14 % - the class rates would differ, making the
    evaluation misleading. stratify= guarantees the same ratio in both halves.

    WHY pass BOTH y_clf and y_reg to train_test_split together?
    scikit-learn applies the same random shuffle to all arrays passed in one
    call. This ensures y_clf_train[i] and y_reg_train[i] describe the
    SAME song (row i of X_train). If we split separately the rows would
    get out of sync.
    """
    print("\n" + "=" * 60)
    print("STEP 5 - TARGETS & TRAIN / TEST SPLIT")
    print("=" * 60)

    y_clf = (df["popularity"] >= POPULARITY_THRESHOLD).astype(int)
    y_reg = df["popularity"].copy().astype(float)
    X = df.drop(columns=["popularity"])

    pos = y_clf.sum()
    print(f"Classification target: {pos:,} popular  /  {len(y_clf)-pos:,} not popular")
    print(f"  Positive class rate: {y_clf.mean() * 100:.1f} %")

    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = (
        train_test_split(
            X, y_clf, y_reg,
            test_size=TEST_SIZE,
            random_state=SEED,
            stratify=y_clf,      # preserve class balance in both halves
        )
    )

    print(f"\nTrain : {len(X_train):,} rows  (positive rate {y_clf_train.mean()*100:.1f} %)")
    print(f"Test  : {len(X_test):,} rows  (positive rate {y_clf_test.mean()*100:.1f} %)")
    return X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test


# =============================================================================
# 6.  PREPROCESSING PIPELINE BUILDER
# =============================================================================
def build_preprocessor(X_sample: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer that handles numeric and categorical columns
    separately. Returns an UNFITTED transformer - it is fitted later inside
    each Pipeline's .fit() call, which is what prevents data leakage.

    Why ColumnTransformer?
    Different columns need different treatment:
    - Numeric columns -> impute median -> scale with RobustScaler
    - Categorical columns (track_genre) -> impute most_frequent -> OrdinalEncoder

    Why RobustScaler instead of StandardScaler?
    StandardScaler centres on the mean and divides by standard deviation.
    Both statistics are sensitive to outliers - one extreme value inflates
    the std and shrinks every other value after scaling. RobustScaler uses
    the median and IQR (interquartile range), which are resistant to
    outliers. Since loudness, duration_ms, and tempo have heavy tails,
    RobustScaler is the safer choice here.

    Why OrdinalEncoder for track_genre?
    We have 114 unique genres. One-hot encoding would produce 114 extra
    columns, making the dataset very wide and causing issues with linear
    models. OrdinalEncoder assigns each genre a number 0-113. Tree-based
    models (Decision Tree, XGBoost) can split on these codes and effectively
    group genres; linear models get less benefit but still receive some signal.
    handle_unknown='use_encoded_value' + unknown_value=-1 means that genres
    never seen during training (e.g., if someone enters a new genre via the
    app) get code -1 instead of causing a crash.

    Why SimpleImputer inside the Pipeline?
    If a test row somehow arrives with a NaN (after deployment, e.g. a user
    leaving a slider blank), the imputer fills it with the training-set median
    before the scaler sees it. It is a safety net.
    """
    numeric_cols     = X_sample.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_sample.select_dtypes(include="object").columns.tolist()

    print(f"\nNumeric features     ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",    # silently ignore any columns not listed above
    )
    return preprocessor


# =============================================================================
# 7.  EVALUATION HELPERS
# =============================================================================
def _evaluate_classifier(name, pipeline, X_train, y_train, X_test, y_test) -> dict:
    """
    Fit the pipeline on training data, then evaluate on the TEST set.

    IMPORTANT: we always report TEST metrics, never train metrics.
    The professor said 'honest evaluation'. A model that memorises training
    data will look great on train but fail on new songs - that is overfitting.
    Test metrics tell you how the model performs on data it has NEVER seen.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    y_prob = None
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]

    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else float("nan"),
        "_pipeline": pipeline,
    }


def _evaluate_regressor(name, pipeline, X_train, y_train, X_test, y_test) -> dict:
    """Fit a regression pipeline and return test-set metrics."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return {
        "Model":   name,
        "R2":      round(r2_score(y_test, y_pred), 4),
        "MAE":     round(mean_absolute_error(y_test, y_pred), 4),
        "RMSE":    round(rmse, 4),
        "_pipeline": pipeline,
        "_y_pred":   y_pred,
        "_y_test":   np.array(y_test),
    }


# =============================================================================
# 8.  CLASSIFICATION MODEL LADDER
# =============================================================================
def train_classification_ladder(X_train, y_train, X_test, y_test) -> tuple:
    """
    Train four classifiers in order of increasing complexity.
    Each model is a complete Pipeline: preprocessor -> (selector) -> model.

    Model 1 - DummyClassifier:
        Always predicts the most common class. This is our FLOOR - if we
        cannot beat this, our model has learned nothing at all.

    Model 2 - Logistic Regression:
        Linear decision boundary. Fast and interpretable. class_weight=
        'balanced' automatically adjusts for our imbalanced classes by
        upweighting the rare 'popular' class during training.
        SelectKBest keeps the 15 most informative features (by ANOVA F-score)
        to avoid noise from less useful columns.

    Model 3 - Decision Tree:
        Non-linear, splits on feature thresholds. Highly interpretable (you
        can literally draw it). Prone to overfitting - we cap max_depth=6 to
        limit how deeply it can memorise training data.

    Model 4 - XGBoost + RandomizedSearchCV:
        Gradient boosting - builds trees sequentially, each correcting the
        errors of the previous one. Usually the best performer.
        RandomizedSearchCV tries n_iter=20 random hyperparameter combinations
        with 3-fold cross-validation, instead of exhaustive grid search which
        would take hours. Tuned on TRAINING data only - test set never seen
        during tuning.
    """
    print("\n" + "=" * 60)
    print("STEP 6a - CLASSIFICATION MODEL LADDER")
    print("=" * 60)
    results = []

    # ── 1. Dummy baseline ─────────────────────────────────────────────────────
    print("\n[1/4] DummyClassifier (most_frequent strategy) ...")
    dummy_clf = Pipeline([
        ("model", DummyClassifier(strategy="most_frequent", random_state=SEED)),
    ])
    results.append(_evaluate_classifier(
        "DummyClassifier", dummy_clf, X_train, y_train, X_test, y_test
    ))
    print(f"      F1 = {results[-1]['F1']:.4f}  |  ROC-AUC = {results[-1]['ROC-AUC']:.4f}")

    # ── 2. Logistic Regression ────────────────────────────────────────────────
    print("\n[2/4] Logistic Regression ...")
    lr_pipe = Pipeline([
        ("pre",    build_preprocessor(X_train)),
        ("select", SelectKBest(f_classif, k=N_SELECT)),
        ("model",  LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=SEED,
        )),
    ])
    results.append(_evaluate_classifier(
        "Logistic Regression", lr_pipe, X_train, y_train, X_test, y_test
    ))
    print(f"      F1 = {results[-1]['F1']:.4f}  |  ROC-AUC = {results[-1]['ROC-AUC']:.4f}")

    # ── 3. Decision Tree ──────────────────────────────────────────────────────
    print("\n[3/4] Decision Tree (max_depth=6) ...")
    dt_pipe = Pipeline([
        ("pre",   build_preprocessor(X_train)),
        ("model", DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=SEED,
        )),
    ])
    results.append(_evaluate_classifier(
        "Decision Tree", dt_pipe, X_train, y_train, X_test, y_test
    ))
    print(f"      F1 = {results[-1]['F1']:.4f}  |  ROC-AUC = {results[-1]['ROC-AUC']:.4f}")

    # ── 4. XGBoost + RandomizedSearchCV ──────────────────────────────────────
    print("\n[4/4] XGBoost + RandomizedSearchCV  (this may take 3-5 minutes) ...")

    # scale_pos_weight tells XGBoost to care more about the rare positive class.
    # Formula: count(negative) / count(positive)
    scale_pos = float((y_train == 0).sum() / (y_train == 1).sum())

    xgb_pipe = Pipeline([
        ("pre",   build_preprocessor(X_train)),
        ("model", XGBClassifier(
            scale_pos_weight=scale_pos,
            random_state=SEED,
            eval_metric="logloss",
            verbosity=0,
        )),
    ])

    param_dist = {
        "model__n_estimators":     [100, 200, 300],
        "model__max_depth":        [3, 4, 5, 6],
        "model__learning_rate":    [0.05, 0.1, 0.2],
        "model__subsample":        [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
        "model__min_child_weight": [1, 3, 5],
    }

    search_clf = RandomizedSearchCV(
        xgb_pipe,
        param_distributions=param_dist,
        n_iter=20,         # try 20 random combinations
        scoring="f1",      # optimise for F1 (better than accuracy on imbalanced data)
        cv=3,              # 3-fold cross-validation on the training set
        random_state=SEED,
        n_jobs=-1,         # use all CPU cores
        verbose=1,
    )
    search_clf.fit(X_train, y_train)
    print(f"\n      Best CV params : {search_clf.best_params_}")

    best_clf = search_clf.best_estimator_
    results.append(_evaluate_classifier(
        "XGBoost", best_clf, X_train, y_train, X_test, y_test
    ))
    print(f"      F1 = {results[-1]['F1']:.4f}  |  ROC-AUC = {results[-1]['ROC-AUC']:.4f}")

    # ── Extra plots for the best model ────────────────────────────────────────
    _plot_confusion_matrix(best_clf, X_test, y_test)
    _plot_roc_curve(best_clf, X_test, y_test)

    return results, best_clf


def _plot_confusion_matrix(pipeline, X_test, y_test) -> None:
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Not popular", "Popular"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title("XGBoost Classifier - Confusion Matrix (test set)")
    plt.tight_layout()
    _save_fig("07_confusion_matrix_xgb.png")


def _plot_roc_curve(pipeline, X_test, y_test) -> None:
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"XGBoost  AUC = {auc:.3f}", color="steelblue", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - XGBoost Classifier (test set)")
    ax.legend()
    plt.tight_layout()
    _save_fig("08_roc_curve_xgb.png")


# =============================================================================
# 9.  REGRESSION MODEL LADDER
# =============================================================================
def train_regression_ladder(X_train, y_train, X_test, y_test) -> tuple:
    """
    Train four regressors in order of increasing complexity.

    Model 1 - DummyRegressor (mean strategy):
        Always predicts the mean popularity score. Again our FLOOR - R2 of 0
        by definition. Any real model must exceed this.

    Model 2 - Ridge Regression:
        Linear with L2 regularisation. The alpha parameter penalises large
        coefficients, shrinking them toward zero. This reduces overfitting
        compared to plain linear regression. SelectKBest selects the top
        features by F-statistic.

    Model 3 - Decision Tree:
        Non-linear regression tree. Same overfitting concern as classifier -
        we cap max_depth=6.

    Model 4 - XGBoost + RandomizedSearchCV:
        Same gradient boosting approach as classification but minimising
        squared error instead of log-loss.
    """
    print("\n" + "=" * 60)
    print("STEP 6b - REGRESSION MODEL LADDER")
    print("=" * 60)
    results = []

    # ── 1. Dummy baseline ─────────────────────────────────────────────────────
    print("\n[1/4] DummyRegressor (mean strategy) ...")
    dummy_reg = Pipeline([
        ("model", DummyRegressor(strategy="mean")),
    ])
    results.append(_evaluate_regressor(
        "DummyRegressor", dummy_reg, X_train, y_train, X_test, y_test
    ))
    print(f"      R2 = {results[-1]['R2']:.4f}  |  RMSE = {results[-1]['RMSE']:.4f}")

    # ── 2. Ridge Regression ───────────────────────────────────────────────────
    print("\n[2/4] Ridge Regression ...")
    ridge_pipe = Pipeline([
        ("pre",    build_preprocessor(X_train)),
        ("select", SelectKBest(f_regression, k=N_SELECT)),
        ("model",  Ridge(alpha=1.0)),
    ])
    results.append(_evaluate_regressor(
        "Ridge Regression", ridge_pipe, X_train, y_train, X_test, y_test
    ))
    print(f"      R2 = {results[-1]['R2']:.4f}  |  RMSE = {results[-1]['RMSE']:.4f}")

    # ── 3. Decision Tree ──────────────────────────────────────────────────────
    print("\n[3/4] Decision Tree (max_depth=6) ...")
    dt_pipe = Pipeline([
        ("pre",   build_preprocessor(X_train)),
        ("model", DecisionTreeRegressor(max_depth=6, random_state=SEED)),
    ])
    results.append(_evaluate_regressor(
        "Decision Tree", dt_pipe, X_train, y_train, X_test, y_test
    ))
    print(f"      R2 = {results[-1]['R2']:.4f}  |  RMSE = {results[-1]['RMSE']:.4f}")

    # ── 4. XGBoost + RandomizedSearchCV ──────────────────────────────────────
    print("\n[4/4] XGBoost + RandomizedSearchCV  (this may take 3-5 minutes) ...")
    xgb_pipe = Pipeline([
        ("pre",   build_preprocessor(X_train)),
        ("model", XGBRegressor(random_state=SEED, verbosity=0)),
    ])

    param_dist = {
        "model__n_estimators":     [100, 200, 300],
        "model__max_depth":        [3, 4, 5, 6],
        "model__learning_rate":    [0.05, 0.1, 0.2],
        "model__subsample":        [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
        "model__min_child_weight": [1, 3, 5],
    }

    search_reg = RandomizedSearchCV(
        xgb_pipe,
        param_distributions=param_dist,
        n_iter=20,
        scoring="r2",
        cv=3,
        random_state=SEED,
        n_jobs=-1,
        verbose=1,
    )
    search_reg.fit(X_train, y_train)
    print(f"\n      Best CV params : {search_reg.best_params_}")

    best_reg = search_reg.best_estimator_
    results.append(_evaluate_regressor(
        "XGBoost", best_reg, X_train, y_train, X_test, y_test
    ))
    print(f"      R2 = {results[-1]['R2']:.4f}  |  RMSE = {results[-1]['RMSE']:.4f}")

    # ── Extra plot for the best model ─────────────────────────────────────────
    _plot_predicted_vs_actual(results[-1]["_y_test"], results[-1]["_y_pred"])

    return results, best_reg


def _plot_predicted_vs_actual(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Scatter plot of predicted vs actual popularity scores.
    Perfect predictions lie on the red diagonal. Spread around it = error.
    """
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y_test), size=min(3_000, len(y_test)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test[idx], y_pred[idx], alpha=0.2, s=8, color="steelblue")
    ax.plot([0, 100], [0, 100], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Popularity")
    ax.set_ylabel("Predicted Popularity")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("XGBoost Regressor - Predicted vs Actual (test set)")
    ax.legend()
    plt.tight_layout()
    _save_fig("09_predicted_vs_actual_xgb.png")


# =============================================================================
# 10. LEADERBOARD
# =============================================================================
def print_leaderboard(clf_results: list, reg_results: list) -> None:
    """
    Print two formatted tables comparing all models on the test set.

    WHY include the Dummy baseline in the table?
    Your professor said: 'honest evaluation'. Showing the baseline next to
    your best model makes it clear how much the ML actually helps, rather
    than just reporting the XGBoost numbers in isolation.
    """
    print("\n" + "=" * 60)
    print("STEP 7 - LEADERBOARD  (all metrics on the HELD-OUT test set)")
    print("=" * 60)

    # Strip private keys (those starting with _) before building the table
    clf_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in clf_results]
    reg_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in reg_results]

    clf_df = pd.DataFrame(clf_rows).set_index("Model")
    reg_df = pd.DataFrame(reg_rows).set_index("Model")

    print("\nCLASSIFICATION  (higher is better for all metrics)")
    print(clf_df.to_string())

    print("\nREGRESSION  (higher R2 is better; lower MAE / RMSE is better)")
    print(reg_df.to_string())


# =============================================================================
# 11. SAVE MODELS
# =============================================================================
def save_models(best_clf, best_reg) -> None:
    """
    Persist the two fully fitted Pipelines to disk.

    WHY save the entire Pipeline and not just the model?
    The Pipeline contains the fitted preprocessor (scaler + encoder) as its
    first step. When api.py loads classifier.pkl and calls
    `pipeline.predict(raw_features)`, the Pipeline automatically:
        1. Scales the numeric features (using the statistics learned from
           the training data)
        2. Encodes track_genre (using the mapping learned from training)
        3. Runs the model
    If we saved only the XGBoost model weights, we would have to manually
    re-apply the exact same scaler in api.py - any mismatch would silently
    produce wrong predictions.

    Professor's test: 'if you shut down the computer, are the parameters
    still there tomorrow?' - YES, because they are on disk in the .pkl file.
    """
    print("\n" + "=" * 60)
    print("STEP 8 - SAVE MODELS  (.pkl files)")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)

    clf_path = os.path.join(MODELS_DIR, "classifier.pkl")
    reg_path = os.path.join(MODELS_DIR, "regressor.pkl")

    joblib.dump(best_clf, clf_path)
    joblib.dump(best_reg, reg_path)

    print(f"Saved -> {clf_path}   ({os.path.getsize(clf_path) / 1e6:.2f} MB)")
    print(f"Saved -> {reg_path}   ({os.path.getsize(reg_path) / 1e6:.2f} MB)")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    print("\n" + "#" * 60)
    print("  SPOTIFY POPULARITY PREDICTOR - OFFLINE TRAINING")
    print("#" * 60)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Step 1 - Load raw data
    df = load_data()

    # Step 2 - EDA (run on raw data before any changes)
    run_eda(df)

    # Step 3 - Clean
    df = clean_data(df)

    # Step 4 - Engineer new features
    df = engineer_features(df)

    # Step 5 - Split into train / test
    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = (
        prepare_and_split(df)
    )

    # Step 6a - Classification model ladder
    clf_results, best_clf = train_classification_ladder(
        X_train, y_clf_train, X_test, y_clf_test
    )

    # Step 6b - Regression model ladder
    reg_results, best_reg = train_regression_ladder(
        X_train, y_reg_train, X_test, y_reg_test
    )

    # Step 7 - Print leaderboard
    print_leaderboard(clf_results, reg_results)

    # Step 8 - Save models
    save_models(best_clf, best_reg)

    print("\n" + "#" * 60)
    print("  TRAINING COMPLETE")
    print("#" * 60)
    print("\nNext steps:")
    print("  Start API   :  uvicorn app.api:app --reload --port 8000")
    print("  Start app   :  streamlit run app/app.py")
    print("  Open docs   :  http://localhost:8000/docs")
    print("  Open app    :  http://localhost:8501")


if __name__ == "__main__":
    main()
