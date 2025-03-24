import json
import matplotlib.pyplot as plt
import seaborn as sns

def plot_logs(log_path, title=None):
    log_data = json.load(open(log_path, 'r'))
    
    if title is None: 
        title = "Loss & Accuracy Evolution"
    
    # Append activated modalities (if present) to the title.
    modalities = log_data.get("modalities", [])
    # if modalities:
    #     title += " (" + ", ".join(modalities) + ")"
    
    for run_key in log_data["runs"].keys():
        _plot_on_run(log_data, run_key, title=title)

def _plot_on_run(log_data, run_key, title):
    steps = log_data["runs"][run_key]["steps"]
    
    # Extract epoch-wise values for main metrics.
    epochs = [step["epoch"] for step in steps]
    train_loss = [step["train"]["loss"] for step in steps]
    validation_loss = [step["val"]["loss"] for step in steps]
    train_acc = [step["train"]["acc"] for step in steps]
    validation_acc = [step["val"]["acc"] for step in steps]
    
    # Plot main Loss and Accuracy together.
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=epochs, y=train_loss, label="Train Loss", marker=".", color="orange", linestyle="--")
    sns.lineplot(x=epochs, y=validation_loss, label="Validation Loss", marker="X", color="orange")
    sns.lineplot(x=epochs, y=train_acc, label="Train Accuracy", marker=".", color="green", linestyle="--")
    sns.lineplot(x=epochs, y=validation_acc, label="Validation Accuracy", marker="X", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Loss & Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()
    
    # Plot sub-loss values separately for Train and Validation.
    if "sub_loss" in steps[0]["train"]:
        modalities = log_data.get("modalities", [])
        
        # Plot training sub-losses.
        plt.figure(figsize=(10, 5))
        for mod in modalities:
            train_sub_loss = [step["train"]["sub_loss"].get(mod, None) for step in steps]
            if any(v is not None for v in train_sub_loss):
                sns.lineplot(x=epochs, y=train_sub_loss, label=f"{mod}", marker=".", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Sub Loss")
        plt.title(title + " - Train Sub Loss per Modality")
        plt.legend()
        plt.grid()
        plt.show()
        
        # Plot validation sub-losses.
        plt.figure(figsize=(10, 5))
        for mod in modalities:
            val_sub_loss = [step["val"]["sub_loss"].get(mod, None) for step in steps]
            if any(v is not None for v in val_sub_loss):
                sns.lineplot(x=epochs, y=val_sub_loss, label=f"{mod}", marker="X")
        plt.xlabel("Epoch")
        plt.ylabel("Sub Loss")
        plt.title(title + " - Validation Sub Loss per Modality")
        plt.legend()
        plt.grid()
        plt.show()
    
    # Plot sub-accuracy values separately for Train and Validation.
    if "sub_acc" in steps[0]["train"]:
        modalities = log_data.get("modalities", [])
        
        # Plot training sub-accuracies.
        plt.figure(figsize=(10, 5))
        for mod in modalities:
            train_sub_acc = [
                sum(step["train"]["sub_acc"].get(mod, [])) / len(step["train"]["sub_acc"].get(mod, [1])) 
                for step in steps
            ]
            sns.lineplot(x=epochs, y=train_sub_acc, label=f"{mod}", marker=".", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Sub Accuracy (avg)")
        plt.title(title + " - Train Sub Accuracy per Modality")
        plt.legend()
        plt.grid()
        plt.show()
        
        # Plot validation sub-accuracies.
        plt.figure(figsize=(10, 5))
        for mod in modalities:
            val_sub_acc = [
                sum(step["val"]["sub_acc"].get(mod, [])) / len(step["val"]["sub_acc"].get(mod, [1])) 
                for step in steps
            ]
            sns.lineplot(x=epochs, y=val_sub_acc, label=f"{mod}", marker="X")
        plt.xlabel("Epoch")
        plt.ylabel("Sub Accuracy (avg)")
        plt.title(title + " - Validation Sub Accuracy per Modality")
        plt.legend()
        plt.grid()
        plt.show()
