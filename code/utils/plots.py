import json
import matplotlib.pyplot as plt
import seaborn as sns

def plot_logs(log_path, title=None):
    log_data = json.load(open(log_path, 'r'))

    if title is None: title = "Loss & Accuracy Evolution"
    _plot_on_run(log_data, list(log_data["runs"].keys())[0], title=title)

def _plot_on_run(log_data, run_key, title):
    steps = log_data["runs"][run_key]["steps"]

    # Extract epoch-wise values
    epochs = [step["epoch"] for step in steps]
    train_loss = [step["train_loss"] for step in steps]
    validation_loss = [step["validation_loss"] for step in steps]
    train_acc = [step["train_acc"] for step in steps]
    validation_acc = [step["validation_acc"] for step in steps]

    # Plot Loss and Accuracy together
    plt.figure(figsize=(10, 5))
    ax1 = sns.lineplot(x=epochs, y=train_loss, label="Train Loss", marker=".", color="orange", linestyle="--")
    sns.lineplot(x=epochs, y=validation_loss, label="Validation Loss", marker="X", color="orange")
    sns.lineplot(x=epochs, y=train_acc, label="Train Accuracy", marker=".", color="green", linestyle="--")
    sns.lineplot(x=epochs, y=validation_acc, label="Validation Accuracy", marker="X", color="green")

    # Labels and title
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss & accuracy")
    plt.title(title)
    
    ax1.legend()
    plt.grid()
    plt.show()
