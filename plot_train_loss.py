import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("wandb_eval_acc_bp_fix_ones_small.csv")

# Identify model names from column headers
model_names = [
    col.replace(" - train/loss", "")
    for col in df.columns
    if col.endswith(" - train/loss")
]


if __name__ == "__main__":
    # Plot
    plt.figure(figsize=(10, 6))
    for model in model_names:
        loss_col = f"{model} - train/loss"
        name, tr_size = model.split("_")[0], model.split("_")[-2][-2:]
        plt.plot(df["Step"], df[loss_col], label=f"{name}, train_len={tr_size}")
        # name, tr_size, test_size = model.split("_")[0], model.split("_")[-2][-2:], model.split("_")[-1][-2:]
        # plt.plot(df["Step"], df[loss_col], label=f"{name}, train_len={tr_size}, eval_len={test_size}")

    plt.title("BP Training Loss with 12 ones")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
