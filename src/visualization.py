###################################
# This script is not used in the main.py file.
# It is used to create visualizations from the results.
###################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import json

def create_visualizations(results_files: List[str], output_dir: str = "results"):
    """Create comprehensive visualizations from results"""
    
    # Combine all results
    all_results = []
    for file in results_files:
        df = pd.read_csv(file)
        # Parse metadata JSON strings
        df["metadata_dict"] = df["metadata"].apply(json.loads)
        all_results.append(df)
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Set style
    plt.style.use('seaborn')
    
    # 1. Single Needle Position Analysis
    plt.figure(figsize=(10, 6))
    single_needle_df = combined_df[
        (combined_df["task_type"] == "single_needle") & 
        (combined_df["metric_type"] == "accuracy")
    ]
    
    sns.barplot(
        data=single_needle_df,
        x="model_name",
        y="value",
        hue=single_needle_df["metadata_dict"].apply(lambda x: x["position"]),
        ci=95
    )
    plt.title("Single Needle Accuracy by Position")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/single_needle_position.png")
    plt.close()
    
    # 2. Multi Needle Performance
    plt.figure(figsize=(12, 6))
    multi_needle_df = combined_df[combined_df["task_type"] == "multi_needle"]
    
    metrics = ["precision", "recall", "f1_score", "partial_retrieval"]
    plt.figure(figsize=(12, 6))
    
    for i, metric in enumerate(metrics):
        metric_data = multi_needle_df[multi_needle_df["metric_type"] == metric]
        plt.subplot(2, 2, i+1)
        sns.boxplot(data=metric_data, x="model_name", y="value")
        plt.title(f"Multi-Needle {metric.replace('_', ' ').title()}")
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/multi_needle_metrics.png")
    plt.close()
    
    # 3. Multi-Hop Success by Hop Count
    plt.figure(figsize=(10, 6))
    multi_hop_df = combined_df[
        (combined_df["task_type"] == "multi_hop") & 
        (combined_df["metric_type"] == "success_rate")
    ]
    
    hop_counts = multi_hop_df["metadata_dict"].apply(lambda x: x["hop_count"])
    sns.lineplot(
        data=multi_hop_df,
        x=hop_counts,
        y="value",
        hue="model_name",
        marker="o"
    )
    plt.title("Multi-Hop Success Rate by Hop Count")
    plt.xlabel("Number of Hops")
    plt.ylabel("Success Rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/multi_hop_success.png")
    plt.close()
    
    # 4. Multi-Hop Error Analysis
    plt.figure(figsize=(8, 6))
    error_df = combined_df[
        (combined_df["task_type"] == "multi_hop") & 
        (combined_df["metric_type"].str.endswith("_error_rate"))
    ]
    
    sns.barplot(
        data=error_df,
        x="model_name",
        y="value",
        hue=error_df["metric_type"].apply(lambda x: x.replace("_error_rate", "")),
        ci=95
    )
    plt.title("Multi-Hop Error Analysis")
    plt.ylabel("Error Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/multi_hop_errors.png")
    plt.close()
    
    # 5. Aggregation Task Performance
    plt.figure(figsize=(10, 6))
    agg_df = combined_df[combined_df["task_type"] == "aggregation"]
    
    metrics = ["completeness", "correctness", "synthesis_quality"]
    for i, metric in enumerate(metrics):
        metric_data = agg_df[agg_df["metric_type"] == metric]
        plt.subplot(1, 3, i+1)
        sns.boxplot(data=metric_data, x="model_name", y="value")
        plt.title(f"Aggregation {metric.title()}")
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/aggregation_performance.png")
    plt.close()
    
    # 6. Document Length Impact
    plt.figure(figsize=(12, 6))
    
    # Calculate success rate for each length bin
    length_impact = combined_df.groupby(
        ["model_name", "task_type", combined_df["metadata_dict"].apply(lambda x: x.get("length_bin", "unknown"))]
    )["value"].mean().reset_index()
    
    sns.barplot(
        data=length_impact,
        x="task_type",
        y="value",
        hue=length_impact["metadata_dict"].apply(lambda x: x.get("length_bin", "unknown")),
        ci=95
    )
    plt.title("Performance by Document Length")
    plt.ylabel("Success Rate")
    plt.xticks(rotation=45)
    plt.legend(title="Document Length")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/length_impact.png")
    plt.close()
    
    # 7. Summary Statistics Table
    summary_stats = pd.DataFrame({
        "Task Type": combined_df["task_type"].unique(),
        "Average Performance": [
            combined_df[combined_df["task_type"] == task]["value"].mean()
            for task in combined_df["task_type"].unique()
        ],
        "95% CI": [
            combined_df[combined_df["task_type"] == task]["value"].std() * 1.96
            for task in combined_df["task_type"].unique()
        ]
    })
    
    summary_stats.to_csv(f"{output_dir}/summary_statistics.csv", index=False)

# Run visualizations
create_visualizations(["results_gpt-4.csv", "results_gpt-4o.csv"])
