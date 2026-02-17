import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
import joblib


def load_data(csv_path: str) -> pd.DataFrame:
    print("🔹 Loading dataset...")
    df = pd.read_csv(csv_path)
    print(f"✅ Dataset loaded with shape: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🔹 Engineering features...")

    # Convert date columns
    df["opened_at"] = pd.to_datetime(
        df["opened_at"], format="%d/%m/%Y %H:%M", errors="coerce"
    )
    df["resolved_at"] = pd.to_datetime(
        df["resolved_at"], format="%d/%m/%Y %H:%M", errors="coerce"
    )
    df["closed_at"] = pd.to_datetime(
        df["closed_at"], format="%d/%m/%Y %H:%M", errors="coerce"
    )

    # Time-based features
    df["opened_hour"] = df["opened_at"].dt.hour
    df["opened_day_of_week"] = df["opened_at"].dt.dayofweek
    df["opened_month"] = df["opened_at"].dt.month
    df["is_weekend"] = df["opened_day_of_week"].isin([5, 6]).astype(int)
    df["is_business_hours"] = (
        (df["opened_hour"] >= 9) & (df["opened_hour"] <= 17)
    ).astype(int)

    # Resolution time (optional feature)
    df["resolution_time_hours"] = (
        df["resolved_at"] - df["opened_at"]
    ).dt.total_seconds() / 3600

    # Numeric priority, impact, urgency
    df["priority_level"] = df["priority"].str.extract(r"(\d+)").astype(float)
    df["impact_level"] = df["impact"].str.extract(r"(\d+)").astype(float)
    df["urgency_level"] = df["urgency"].str.extract(r"(\d+)").astype(float)

    # Targets
    df["sla_breach"] = (~df["made_sla"]).astype(int)
    df["priority_class"] = df["priority"]

    print("✅ Feature engineering complete.")
    return df


def prepare_model_data(df: pd.DataFrame):
    print("\n🔹 Preparing data for modeling...")

    feature_columns = [
        "reassignment_count",
        "reopen_count",
        "sys_mod_count",
        "impact_level",
        "urgency_level",
        "opened_hour",
        "opened_day_of_week",
        "opened_month",
        "is_weekend",
        "is_business_hours",
        "category",
        "subcategory",
        "contact_type",
        "location",
        "assignment_group",
    ]

    df_model = df[feature_columns + ["sla_breach", "priority_class"]].copy()

    # Encode categorical features
    categorical_features = [
        "category",
        "subcategory",
        "contact_type",
        "location",
        "assignment_group",
    ]
    le_dict = {}

    for col in categorical_features:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        le_dict[col] = le

    X = df_model.drop(["sla_breach", "priority_class"], axis=1)
    y_sla = df_model["sla_breach"]
    y_priority = df_model["priority_class"]

    X_train, X_test, y_sla_train, y_sla_test = train_test_split(
        X,
        y_sla,
        test_size=0.2,
        random_state=42,
        stratify=y_sla,
    )

    # Separate split for priority (same X split reused)
    _, _, y_priority_train, y_priority_test = train_test_split(
        X,
        y_priority,
        test_size=0.2,
        random_state=42,
        stratify=y_priority,
    )

    print("✅ Data preparation complete.")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    return (
        X_train,
        X_test,
        y_sla_train,
        y_sla_test,
        y_priority_train,
        y_priority_test,
        le_dict,
        feature_columns,
    )


def train_sla_model(X_train, X_test, y_train, y_test):
    print("\n🚀 Training Model 1: SLA Breach Prediction (Random Forest)...")

    rf_sla = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    rf_sla.fit(X_train, y_train)
    y_pred = rf_sla.predict(X_test)
    y_proba = rf_sla.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("\n✅ SLA model trained.")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["SLA Met", "SLA Breached"]))

    feature_importance = pd.DataFrame(
        {"feature": X_train.columns, "importance": rf_sla.feature_importances_}
    ).sort_values("importance", ascending=False)

    return rf_sla, {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "feature_importance": feature_importance,
    }


def train_priority_model(X_train, X_test, y_train, y_test):
    print("\n🚀 Training Model 2: Priority Classification (Random Forest)...")

    rf_priority = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    rf_priority.fit(X_train, y_train)
    y_pred = rf_priority.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n✅ Priority model trained.")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    feature_importance = pd.DataFrame(
        {"feature": X_train.columns, "importance": rf_priority.feature_importances_}
    ).sort_values("importance", ascending=False)

    # Per-class simple accuracy (optional)
    classes = sorted(y_test.unique())
    print("\nPer-class accuracy:")
    for cls in classes:
        mask = y_test == cls
        if mask.sum() > 0:
            cls_acc = (y_pred[mask] == cls).mean()
            print(f"{cls}: {cls_acc * 100:.2f}% (n={mask.sum()})")

    return rf_priority, {
        "accuracy": accuracy,
        "feature_importance": feature_importance,
    }


def save_artifacts(
    sla_model,
    priority_model,
    le_dict,
    feature_columns,
    sla_metrics,
    priority_metrics,
):
    print("\n🔹 Saving models and artifacts...")

    artifacts = {
        "sla_model": sla_model,
        "priority_model": priority_model,
        "label_encoders": le_dict,
        "feature_columns": feature_columns,
        "sla_metrics": {
            "accuracy": sla_metrics["accuracy"],
            "precision": sla_metrics["precision"],
            "recall": sla_metrics["recall"],
            "f1": sla_metrics["f1"],
            "auc": sla_metrics["auc"],
        },
        "priority_metrics": {
            "accuracy": priority_metrics["accuracy"],
        },
    }

    joblib.dump(artifacts, "incident_ai_models.pkl")
    sla_metrics["feature_importance"].to_csv("sla_feature_importance.csv", index=False)
    priority_metrics["feature_importance"].to_csv(
        "priority_feature_importance.csv", index=False
    )

    print("✅ Saved:")
    print("- incident_ai_models.pkl")
    print("- sla_feature_importance.csv")
    print("- priority_feature_importance.csv")


def main():
    # 1. Load data
    df = load_data("incident_event_log.csv")

    # 2. Feature engineering
    df = engineer_features(df)

    # 3. Prepare model data
    (
        X_train,
        X_test,
        y_sla_train,
        y_sla_test,
        y_priority_train,
        y_priority_test,
        le_dict,
        feature_columns,
    ) = prepare_model_data(df)

    # 4. Train models
    sla_model, sla_metrics = train_sla_model(X_train, X_test, y_sla_train, y_sla_test)
    priority_model, priority_metrics = train_priority_model(
        X_train, X_test, y_priority_train, y_priority_test
    )

    # 5. Save artifacts
    save_artifacts(
        sla_model,
        priority_model,
        le_dict,
        feature_columns,
        sla_metrics,
        priority_metrics,
    )

    print("\n🎉 All done. Models trained and artifacts saved.")


if __name__ == "__main__":
    main()
