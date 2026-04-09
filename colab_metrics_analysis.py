from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scoring.evaluator import evaluate_answer


REQUIRED_COLUMNS = {"student_answer", "reference_answer"}
OPTIONAL_COLUMNS = {"question_text", "actual_marks", "max_marks"}


def evaluate_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows: list[dict] = []
    for _, row in dataframe.fillna("").iterrows():
        result = evaluate_answer(
            student_text=str(row["student_answer"]),
            reference_text=str(row["reference_answer"]),
            question_text=str(row.get("question_text", "")),
            max_marks=int(row.get("max_marks", 10) or 10),
        )

        rows.append(
            {
                **row.to_dict(),
                "semantic_similarity": result["semantic_similarity"],
                "keyword_coverage": result["keyword_match_score"],
                "lexical_similarity": result["lexical_similarity"],
                "question_alignment": result["question_alignment"],
                "grammar_score": result["grammar_score"],
                "length_penalty_factor": result["length_penalty_factor"],
                "predicted_marks": result["final_marks"],
                "matched_keywords": ", ".join(result["matched_keywords"]),
                "missing_keywords": ", ".join(result["missing_keywords"]),
            }
        )

    return pd.DataFrame(rows)


def summarize_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = {
        "mean_semantic_similarity": results_df["semantic_similarity"].mean(),
        "mean_keyword_coverage": results_df["keyword_coverage"].mean(),
        "mean_lexical_similarity": results_df["lexical_similarity"].mean(),
        "mean_question_alignment": results_df["question_alignment"].mean(),
        "mean_grammar_score": results_df["grammar_score"].mean(),
        "mean_length_penalty_factor": results_df["length_penalty_factor"].mean(),
        "mean_predicted_marks": results_df["predicted_marks"].mean(),
    }

    if "actual_marks" in results_df.columns:
        valid = results_df["actual_marks"].astype(str).str.len() > 0
        benchmark_df = results_df.loc[valid].copy()
        if not benchmark_df.empty:
            benchmark_df["actual_marks"] = benchmark_df["actual_marks"].astype(float)
            mae = mean_absolute_error(benchmark_df["actual_marks"], benchmark_df["predicted_marks"])
            rmse = mean_squared_error(
                benchmark_df["actual_marks"],
                benchmark_df["predicted_marks"],
                squared=False,
            )
            corr = benchmark_df["actual_marks"].corr(benchmark_df["predicted_marks"])
            within_1 = (
                (benchmark_df["actual_marks"] - benchmark_df["predicted_marks"]).abs() <= 1
            ).mean() * 100

            summary.update(
                {
                    "mae": mae,
                    "rmse": rmse,
                    "pearson_correlation": corr,
                    "within_1_mark_accuracy": within_1,
                }
            )

    return pd.DataFrame(
        [{"metric": key, "value": round(float(value), 4)} for key, value in summary.items()]
    )


def plot_metric_bars(results_df: pd.DataFrame) -> None:
    metrics = [
        "semantic_similarity",
        "keyword_coverage",
        "lexical_similarity",
        "question_alignment",
        "grammar_score",
    ]
    means = results_df[metrics].mean()

    plt.figure(figsize=(10, 5))
    plt.bar(means.index, means.values, color=["#2563eb", "#16a34a", "#f59e0b", "#7c3aed", "#ef4444"])
    plt.ylabel("Average Score")
    plt.ylim(0, 100)
    plt.title("Average Evaluation Metrics")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()


def plot_marks_scatter(results_df: pd.DataFrame) -> None:
    if "actual_marks" not in results_df.columns:
        return

    valid = results_df["actual_marks"].astype(str).str.len() > 0
    benchmark_df = results_df.loc[valid].copy()
    if benchmark_df.empty:
        return

    benchmark_df["actual_marks"] = benchmark_df["actual_marks"].astype(float)

    plt.figure(figsize=(6, 6))
    plt.scatter(benchmark_df["actual_marks"], benchmark_df["predicted_marks"], color="#2563eb", alpha=0.8)
    max_score = max(benchmark_df["actual_marks"].max(), benchmark_df["predicted_marks"].max())
    plt.plot([0, max_score], [0, max_score], "--", color="#111827")
    plt.xlabel("Actual Marks")
    plt.ylabel("Predicted Marks")
    plt.title("Predicted vs Actual Marks")
    plt.tight_layout()
    plt.show()


def plot_marks_distribution(results_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(results_df["predicted_marks"], bins=10, color="#0f766e", edgecolor="white")
    plt.xlabel("Predicted Marks")
    plt.ylabel("Count")
    plt.title("Distribution of Predicted Marks")
    plt.tight_layout()
    plt.show()


def plot_error_distribution(results_df: pd.DataFrame) -> None:
    if "actual_marks" not in results_df.columns:
        return

    valid = results_df["actual_marks"].astype(str).str.len() > 0
    benchmark_df = results_df.loc[valid].copy()
    if benchmark_df.empty:
        return

    benchmark_df["actual_marks"] = benchmark_df["actual_marks"].astype(float)
    errors = benchmark_df["predicted_marks"] - benchmark_df["actual_marks"]

    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=10, color="#dc2626", edgecolor="white")
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.title("Prediction Error Distribution")
    plt.tight_layout()
    plt.show()


def run_analysis(csv_path: str | Path, output_csv_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_df = pd.read_csv(csv_path)
    results_df = evaluate_dataset(source_df)
    summary_df = summarize_metrics(results_df)

    if output_csv_path:
        results_df.to_csv(output_csv_path, index=False)

    print("Dataset summary")
    print(summary_df.to_string(index=False))

    plot_metric_bars(results_df)
    plot_marks_distribution(results_df)
    plot_marks_scatter(results_df)
    plot_error_distribution(results_df)

    return results_df, summary_df


def example_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "question_text": "Explain stemming and lemmatization in NLP.",
                "reference_answer": (
                    "Stemming reduces words to a root-like form, while lemmatization maps "
                    "words to their dictionary base form using context."
                ),
                "student_answer": (
                    "Stemming removes suffixes to get root forms and lemmatization returns "
                    "dictionary forms using context."
                ),
                "actual_marks": 8,
                "max_marks": 10,
            },
            {
                "question_text": "What is overfitting in machine learning?",
                "reference_answer": (
                    "Overfitting happens when a model memorizes training data and performs "
                    "poorly on unseen data."
                ),
                "student_answer": (
                    "Overfitting means the model fits the training data too closely and does "
                    "not generalize well to new data."
                ),
                "actual_marks": 8,
                "max_marks": 10,
            },
        ]
    )


def save_example_csv(path: str | Path = "sample_exam_answers.csv") -> Path:
    output_path = Path(path)
    example_dataframe().to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    sample_path = save_example_csv()
    run_analysis(sample_path, "sample_exam_results.csv")
