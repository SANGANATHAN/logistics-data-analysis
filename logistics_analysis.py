"""
Week 1 - Logistics Data Analyst Intern
Project: Delivery Route Optimization Using Data Analytics

This script performs basic logistics data exploration and KPI analysis.
The dataset is a synthetic sample created for learning/internship submission.
"""

from pathlib import Path
import pandas as pd

DATA_FILE = Path(__file__).with_name("logistics_data.csv")


def load_data(file_path: Path) -> pd.DataFrame:
    """Load the logistics CSV file."""
    return pd.read_csv(file_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean duplicate rows and handle missing numeric values."""
    df = df.drop_duplicates().copy()

    numeric_columns = [
        "distance_km",
        "delivery_time_min",
        "expected_time_min",
        "fuel_consumed_l",
        "transport_cost_inr",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].fillna(df[column].median())

    df["on_time"] = pd.to_numeric(df["on_time"], errors="coerce").fillna(0).astype(int)
    return df


def calculate_kpis(df: pd.DataFrame) -> dict:
    """Calculate the main KPIs requested in the Week 1 task."""
    on_time_rate = df["on_time"].mean() * 100

    return {
        "total_orders": len(df),
        "on_time_delivery_rate_percent": round(on_time_rate, 2),
        "average_delivery_time_min": round(df["delivery_time_min"].mean(), 2),
        "average_distance_km": round(df["distance_km"].mean(), 2),
        "average_fuel_consumed_l": round(df["fuel_consumed_l"].mean(), 2),
        "average_transport_cost_inr": round(df["transport_cost_inr"].mean(), 2),
    }


def route_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize delivery performance by destination."""
    summary = (
        df.groupby("destination")
        .agg(
            orders=("order_id", "count"),
            avg_distance_km=("distance_km", "mean"),
            avg_delivery_time_min=("delivery_time_min", "mean"),
            avg_cost_inr=("transport_cost_inr", "mean"),
            on_time_rate=("on_time", "mean"),
        )
        .reset_index()
    )

    summary["on_time_rate_percent"] = (summary["on_time_rate"] * 100).round(2)
    summary = summary.drop(columns="on_time_rate")

    return summary.sort_values("avg_cost_inr", ascending=False)


def main() -> None:
    data = load_data(DATA_FILE)

    print("\n--- DATA PREVIEW ---")
    print(data.head())

    print("\n--- DATA INFORMATION ---")
    print(data.info())

    print("\n--- MISSING VALUES ---")
    print(data.isnull().sum())

    data = clean_data(data)
    kpis = calculate_kpis(data)

    print("\n--- LOGISTICS KPIs ---")
    for name, value in kpis.items():
        print(f"{name}: {value}")

    print("\n--- DESTINATION / ROUTE SUMMARY ---")
    print(route_summary(data).to_string(index=False))

    # Simple analytical insight
    delayed = data[data["on_time"] == 0]
    if not delayed.empty:
        print(
            f"\nDelayed deliveries: {len(delayed)} "
            f"({len(delayed) / len(data) * 100:.2f}%)"
        )

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
