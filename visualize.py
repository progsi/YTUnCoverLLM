from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

METRICS = ['WoA_spurious', 'WoA_incorrect', 'WoA_missed', 
               'Artist_spurious', 'Artist_incorrect', 'Artist_missed']
    
def heatmaps(data: pd.DataFrame, metrics: List[str]):
    """Generates plot of multiple heatmaps for the different metrics.
    Args:
        data (pd.DataFrame): dataframe
    """

    def get_metric_sums(df, metric):
        metric_df = df.xs(metric, axis=1, level=2)
        metric_sum = metric_df.sum()
        metric_sum_df = metric_sum.unstack(level=1)
        metric_sum_df = metric_sum_df[["fmtcorrect", "fmtfailed", "postcutoff"]]
        metric_sum_df.columns = ["FMT Passed", "FMT Failed", "Post-Cutoff"]
        metric_sum_df.index = [["None", "Lvl. 1", "Lvl. 2"]]
        return metric_sum_df

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.92, .3, .03, .4])

    N = data[("", "", "TEXT_template")].apply(str).nunique()

    for i, metric in enumerate(metrics):
        plot_data = get_metric_sums(data, metric).apply(lambda x: round(x/N, 2))
        ax = axs[i // 3, i % 3]
        sns.heatmap(plot_data, annot=True, fmt="g", cmap="Blues", ax=ax, 
                    cbar=(i == 0), cbar_ax=None if i else cbar_ax, annot_kws={"size": 21})  # Increase annotation font size
        
        ax.set_title(metric.replace("_", " "), fontsize=21)
        
        if i % 3 == 0:
            ax.set_ylabel("Perturbation", fontsize=19)
        else:
            ax.set_ylabel("")
        # Increase font size of x-ticks and y-ticks
        ax.tick_params(axis='x', labelsize=21, rotation=25)  # X-axis tick font size
        ax.tick_params(axis='y', labelsize=21)  # Y-axis tick font size

    # Set colorbar label size
    cbar_ax.tick_params(labelsize=19)

    plt.tight_layout(rect=[0, 0, .9, 1])
    plt.savefig("heatmaps.pdf")
    plt.show()

def cdf(data: pd.DataFrame):
    """Generates plot of CDF for F1 scores.
    Args:
        data (pd.DataFrame): dataframe
    """
    data[("", "", "f1")] = data[('', 'postcutoff','metrics_overall')].apply(lambda x: x["f1"] if x else None)
    sns.ecdfplot(data, x=("", "", "f1"), hue=("", "", "Origin"), legend="Data Source")
    legend = plt.gca().get_legend()
    legend.set_title("Data Source")
    plt.xlabel("F1")
    plt.savefig("cdf.pdf")
    plt.show()
    
if __name__ == "__main__":
    data = pd.read_parquet("data.parquet")
    heatmaps(data, metrics=METRICS)
    cdf(data)
    