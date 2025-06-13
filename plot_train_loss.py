import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("dyck_3232,4848_recall.csv")

# Identify model names from column headers
graph_type = "eval/recall"
model_names = [
    col.replace(f" - {graph_type}", "")
    for col in df.columns
    if col.endswith(f" - {graph_type}")
]


if __name__ == "__main__":
    # colors = ["#1f77b4", "#aec7e8", "#9467bd", "#c5b0d5"]
    # Plot
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(model_names):
        loss_col = f"{model} - {graph_type}"
        name, tr_size = model.split("_")[0], model.split("_")[-2][-2:]
        eval_size = model.split("_")[-1][-2:]
        plt.plot(df["Step"], df[loss_col]*100, label=f"{name}")
        # name, tr_size, test_size = model.split("_")[0], model.split("_")[-2][-2:], model.split("_")[-1][-2:]
        # plt.plot(df["Step"], df[loss_col], label=f"{name}, train_len={tr_size}, eval_len={test_size}")

    # plt.title("BP Eval Accuracy with 12 ones")
    plt.xlabel("Step")
    plt.ylabel("Recall")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
