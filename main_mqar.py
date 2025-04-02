import argparse
import os
import torch

from mqar.mqar_dataset import MQARDatasetIterator
from mqar.mqar_trainer import MQARTrainer
from mqar.mqar_modelgpt import Model
from mqar.mqar_modelgpt import DEFAULT_CONFIG as MODEL_DEFAULT_CONFIG

SEQ_LEN = 10 #MODEL_DEFAULT_CONFIG["seq_len"]

def run(device):
    ## Setup Dataset
    n_values = 20
    n_keys = 3 
    train_ds = MQARDatasetIterator(
        batch_size=32,
        num_pairs=SEQ_LEN // 2,
        n_keys=n_keys,
        n_values=n_values,
        unique_keys=False,
        all_queries_for_input=False,
        device=device,
    )

    eval_ds = MQARDatasetIterator(
        batch_size=32,
        num_pairs=SEQ_LEN // 2,
        n_keys=n_keys,
        n_values=n_values,
        unique_keys=False,
        all_queries_for_input=False,
        device=device,
    )


    ## Setup Model
    config = MODEL_DEFAULT_CONFIG
    config["device"] = device
    config["seq_len"] = SEQ_LEN + 1 #config["seq_len"]
    config["value_size"] = n_values  
    config["key_size"] = n_keys 
    config["vocab_size"] = n_values + n_keys + 1

    model = Model(config=config).to(device)

    ## Setup Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    ## Train
    trainer = MQARTrainer(
        model=model,
        train_loader=train_ds,
        eval_loader=eval_ds,
        optimizer=optimizer,
        device=device,
        max_steps=50000,
        eval_every=500,
    )

    print("Begin training ... ")
    trainer.train()
    print("Done!")

def main():
    """Runs a Languini experiment using a GPT model."""

    parser = argparse.ArgumentParser(description="Languini Experiment")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu", "mps"],
        default="cpu",
        help="Device to use for training",
    )
    args = parser.parse_args()

    if args.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Languini Experiment")

    # load the config file
    project_path = os.path.dirname(os.path.abspath(__file__))
    print(f"project path: {project_path}")

    run(device)


if __name__ == "__main__":
    main()