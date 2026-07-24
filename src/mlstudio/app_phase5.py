"""

An example of a supervised regression case.
This app is used to verify project workflow.

Author: Tiffany Hildebrand
Date: 2026-06

Process:
    - Load a CSV dataset.
    - Train a supervised regression model.
    - Evaluate model performance.
    - Predict one new case.
    - Create useful charts.

Data Source:
- data/raw/hours_scores_case.csv

Terminal command to run this file from the root project folder:

uv run python -m mlstudio.app_case

OBS:
  Don't edit this file - it should remain a working example.
  It is used in each module to test the installation and workflow.
  You never need to do anything with it, but if would like,
  you can copy it, rename it, and modify your copy.
  If you do, include your command to run it in the docstring above and in README.md.
"""

# === Section 1a. DECLARE IMPORTS ===

import logging
from typing import Final

from datafun_toolkit.logger import get_logger, log_header
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# === Section 1b. CONFIGURE LOGGER ONCE PER MODULE ===

LOG: logging.Logger = get_logger("ML", level="DEBUG")
log_header(LOG, "ML")

# === Section 1c. Global Constants and Configuration ===

DATASET_NAME: Final[str] = "diabetes"

# We are predicting BMI, which is a continuous numeric value.
TARGET_COL: Final[str] = "BMI"

# Health and demographic features used to predict BMI.
FEATURE_COLS: Final[list[str]] = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]

# Hold back 20% of the data for testing.
TEST_SIZE: Final[float] = 0.20

# Keep the split reproducible.
RANDOM_STATE: Final[int] = 42

# === Section 1d. Pandas Configuration for Display ===

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)


# === Section 2. Load the Data ===


def load_data() -> pd.DataFrame:
    """Load the diabetes health indicators dataset."""
    LOG.info(f"Loading dataset: {DATASET_NAME}")

    df: pd.DataFrame = pd.read_csv(f"data/raw/{DATASET_NAME}.csv")

    LOG.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    LOG.debug(f"\n{df.head()}")

    return df


# === Section 3. Inspect Data Shape and Structure ===


def inspect_basic(df: pd.DataFrame) -> None:
    """Inspect basic dataset structure."""
    LOG.info("Column names")
    LOG.debug(f"{list(df.columns)}")

    LOG.info("DataFrame info")
    df.info()

    LOG.info(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")


# === Section 4. Check Data Quality ===


def check_quality(df: pd.DataFrame) -> None:
    """Check missing values and duplicate rows."""
    LOG.info("Missing values by column")
    LOG.debug(f"\n{df.isna().sum()}")

    duplicate_count: int = df.duplicated().sum()
    LOG.info(f"Duplicate row count: {duplicate_count}")


# === Section 5. Create a Clean View ===


def make_clean_view(df: pd.DataFrame) -> pd.DataFrame:
    """Create a clean modeling view with selected features and target."""
    LOG.info("Creating clean modeling view")

    selected_cols: list[str] = FEATURE_COLS + [TARGET_COL]

    # Select only the columns needed for the regression model.
    df_selected: pd.DataFrame = df[selected_cols]

    # Remove rows containing missing values.
    df_no_missing: pd.DataFrame = df_selected.dropna()

    # Make a copy to avoid SettingWithCopyWarning.
    df_clean: pd.DataFrame = df_no_missing.copy()

    LOG.info(f"Clean view: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")

    return df_clean


# === Section 6. Train Supervised Model ===


def train_model(
    df_clean: pd.DataFrame,
) -> tuple[LinearRegression, pd.Series, pd.Series]:
    """Train and evaluate a linear regression model."""
    LOG.info("Training LinearRegression model")

    x = df_clean[FEATURE_COLS]
    y = df_clean[TARGET_COL]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = pd.Series(
        model.predict(x_test),
        index=y_test.index,
    )

    mae: float = mean_absolute_error(y_test, y_pred)
    rmse: float = mean_squared_error(y_test, y_pred) ** 0.5
    r2: float = r2_score(y_test, y_pred)

    LOG.info(f"Mean absolute error (MAE): {mae:.2f}")
    LOG.info(f"Root mean squared error (RMSE): {rmse:.2f}")
    LOG.info(f"R-squared: {r2:.2f}")

    return model, y_test, y_pred


# === Section 7. Predict One New Case ===


def predict_example(model: LinearRegression) -> None:
    """Use the trained model to predict BMI for one new case."""
    LOG.info("Predicting BMI for one new case")

    new_case = pd.DataFrame(
        [
            {
                "HighBP": 1.0,
                "HighChol": 1.0,
                "CholCheck": 1.0,
                "Smoker": 0.0,
                "Stroke": 0.0,
                "HeartDiseaseorAttack": 0.0,
                "PhysActivity": 1.0,
                "Fruits": 1.0,
                "Veggies": 1.0,
                "HvyAlcoholConsump": 0.0,
                "AnyHealthcare": 1.0,
                "NoDocbcCost": 0.0,
                "GenHlth": 3.0,
                "MentHlth": 2.0,
                "PhysHlth": 2.0,
                "DiffWalk": 0.0,
                "Sex": 1.0,
                "Age": 8.0,
                "Education": 5.0,
                "Income": 7.0,
            }
        ]
    )

    predicted_bmi: float = model.predict(new_case)[0]

    LOG.info(f"New case:\n{new_case}")
    LOG.info(f"Predicted BMI: {predicted_bmi:.1f}")


# === Section 8. Create Visualizations ===


def make_plots(
    y_test: pd.Series,
    y_pred: pd.Series,
) -> None:
    """Create custom charts for actual versus predicted BMI."""

    # === Chart 1: Actual BMI vs Predicted BMI ===

    LOG.info("Creating chart: actual BMI vs predicted BMI")

    fig, ax = plt.subplots(figsize=(9, 5))

    scatter_plt: Axes = sns.scatterplot(
        x=y_test,
        y=y_pred,
        ax=ax,
    )

    # Add a reference line showing perfect predictions.
    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())

    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
    )

    scatter_plt.set_title("Actual BMI vs Predicted BMI (CLOSE chart to continue)")
    scatter_plt.set_xlabel("Actual BMI")
    scatter_plt.set_ylabel("Predicted BMI")

    # === Chart 2: Residual Distribution ===

    LOG.info("Creating chart: residual distribution")

    residuals = y_test - y_pred

    fig, ax = plt.subplots(figsize=(9, 5))

    residual_plt: Axes = sns.histplot(
        residuals,
        kde=True,
        ax=ax,
    )

    residual_plt.set_title(
        "Distribution of BMI Prediction Errors (CLOSE chart to continue)"
    )
    residual_plt.set_xlabel("Residual: Actual BMI - Predicted BMI")
    residual_plt.set_ylabel("Count")


# === Section 9. Summary and Next Steps ===


def summarize(
    df: pd.DataFrame,
    df_clean: pd.DataFrame,
) -> None:
    """Log a brief summary of the custom regression project."""
    LOG.info("========================")
    LOG.info("SUMMARY")
    LOG.info("========================")
    LOG.info(f"Dataset: {DATASET_NAME}")
    LOG.info(f"Original rows: {df.shape[0]}")
    LOG.info(f"Clean rows: {df_clean.shape[0]}")
    LOG.info(f"Features: {FEATURE_COLS}")
    LOG.info(f"Target: {TARGET_COL}")


# === DEFINE THE MAIN FUNCTION ===


def main() -> None:
    """Run the custom diabetes BMI regression workflow."""
    log_header(LOG, "ML")

    LOG.info("========================")
    LOG.info("START main()")
    LOG.info("========================")

    LOG.info("Load dataset..............")
    df = load_data()

    LOG.info("Inspect dataset...........")
    inspect_basic(df)

    LOG.info("Check data quality........")
    check_quality(df)

    LOG.info("Create clean view.........")
    df_clean = make_clean_view(df)

    LOG.info("Train supervised model....")
    model, y_test, y_pred = train_model(df_clean)

    LOG.info("Predict one case..........")
    predict_example(model)

    LOG.info("Create charts.............")
    make_plots(y_test, y_pred)

    LOG.info("Summarize workflow........")
    summarize(df, df_clean)

    LOG.info(
        "----- in a script, call plt.show() once at the end to display all charts -----"
    )
    LOG.info(
        "----- in a script, CLOSE the chart windows with the close button to CONTINUE -----"
    )

    plt.show()

    LOG.info("Workflow complete")
    LOG.info("IMPORTANT: This script creates chart windows.")
    LOG.info("Close chart windows and terminate this process with CTRL+c as needed.")
    LOG.info("========================")
    LOG.info("Executed successfully!")
    LOG.info("========================")


# === CONDITIONAL EXECUTION GUARD ===

if __name__ == "__main__":
    main()
