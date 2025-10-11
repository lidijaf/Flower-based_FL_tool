import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
from ruamel.yaml import YAML

# --- Setup paths and YAML ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
yaml = YAML()

base_dir = os.path.dirname(__file__)
conf_dir = os.path.join(base_dir, "conf")
config_common = os.path.join(conf_dir, "config_common.yaml")
config_client = os.path.join(conf_dir, "config_client.yaml")
config_server = os.path.join(conf_dir, "config_server.yaml")

# --- Default user selections ---
choices = {
    "task": "",
    "dataset": "",
    "model": "",
    "algorithm": ""
}

additional_settings = {
    "num_rounds": "int",
    "num_clients": "int",
    "min_available_clients": "int",
    "min_fit_clients": "int",
    "min_evaluate_clients": "int",
    "epochs": "int",
    "batch_size": "int"
}

# --- Utility functions ---
def update_yaml(filepath, updates: dict):
    """Safely update YAML files while keeping existing keys."""
    if not os.path.exists(filepath):
        print(f"[INFO] Creating new config file: {filepath}")
        config = {}
    else:
        with open(filepath, "r") as f:
            config = yaml.load(f) or {}

    config.update(updates)
    with open(filepath, "w") as f:
        yaml.dump(config, f)
    print(f" Updated {os.path.basename(filepath)} with {updates}")


# --- GUI Setup ---
root = tk.Tk()
root.title("Federated Learning Simulation")
root.geometry('700x500')

frm = ttk.Frame(root, padding=10)
frm.grid()


# --- Task Selection ---
def show_task_options():
    ttk.Label(frm, text="Please select the task of interest:").grid(column=0, row=0)
    task_options = ["classification", "anomaly detection"]
    value_inside = tk.StringVar(value="Select an Option")
    menu = ttk.OptionMenu(frm, value_inside, "Select an Option", *task_options)
    menu.grid(column=0, row=1)

    def on_submit():
        task = value_inside.get()
        choices["task"] = task
        update_yaml(config_common, {"task": task})
        print(f"Selected Task: {task}")
        show_dataset_options()

    ttk.Button(frm, text='Submit', command=on_submit).grid(column=0, row=2)


# --- Dataset Selection ---
def show_dataset_options():
    dataset_options = []
    if choices["task"] == "classification":
        dataset_options = ["mnist", "cifar10", "cifar100", "fmnist", "your_dataset"]
    elif choices["task"] == "anomaly detection":
        dataset_options = ["metro", "your_dataset"]

    ttk.Label(frm, text="Please select the dataset:").grid(column=0, row=3)
    value_inside = tk.StringVar(value="Select an Option")
    menu = ttk.OptionMenu(frm, value_inside, "Select an Option", *dataset_options)
    menu.grid(column=0, row=4)

    def on_submit():
        dataset = value_inside.get()
        choices["dataset"] = dataset
        update_yaml(config_common, {"dataset": dataset})
        print(f"Selected Dataset: {dataset}")
        show_model_options()

    ttk.Button(frm, text='Submit', command=on_submit).grid(column=0, row=5)


# --- Model Selection ---
def show_model_options():
    if choices["dataset"] == "mnist":
        model_options = ["CNN_MNIST", "your_model"]
    elif choices["dataset"] == "cifar10":
        model_options = ["CNN_CIFAR10", "ResNet18", "your_model"]
    elif choices["dataset"] == "fmnist":
        model_options = ["CNN_FMNIST", "your_model"]
    elif choices["dataset"] == "cifar100":
        model_options = ["CNN_CIFAR100", "your_model"]
    elif choices["dataset"] in ["metro", "your_dataset"]:
        model_options = ["Autoencoder", "Transformer", "your_model"]
    else:
        model_options = ["your_model"]

    ttk.Label(frm, text="Please select the model:").grid(column=0, row=6)
    value_inside = tk.StringVar(value="Select an Option")
    menu = ttk.OptionMenu(frm, value_inside, "Select an Option", *model_options)
    menu.grid(column=0, row=7)

    def on_submit():
        model = value_inside.get()
        choices["model"] = model
        update_yaml(config_common, {"model": model})
        print(f"Selected Model: {model}")
        set_training_algorithm()

    ttk.Button(frm, text='Submit', command=on_submit).grid(column=0, row=8)


# --- Algorithm Selection ---
def set_training_algorithm():
    ttk.Label(frm, text="Please select the FL algorithm:").grid(column=0, row=9)
    algorithm_options = ["fedavg", "pfedme"]
    value_inside = tk.StringVar(value="Select an Option")
    menu = ttk.OptionMenu(frm, value_inside, "Select an Option", *algorithm_options)
    menu.grid(column=0, row=10)

    def on_submit():
        algorithm = value_inside.get()
        choices["algorithm"] = algorithm
        update_yaml(config_common, {"algorithm": algorithm})
        print(f"Selected Algorithm: {algorithm}")
        enable_final_button()

    ttk.Button(frm, text='Submit', command=on_submit).grid(column=0, row=11)


# --- Additional Settings ---
def enable_final_button():
    def open_additional_settings():
        root2 = tk.Toplevel(root)
        root2.title("Additional Settings")
        root2.geometry('900x700')

        frame2 = ttk.Frame(root2, padding="10")
        frame2.grid(row=0, column=0, sticky="NSEW")

        ttk.Label(frame2, text="Parameter").grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(frame2, text="Type").grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(frame2, text="Value").grid(row=0, column=2, padx=5, pady=5)

        input_fields = []
        for i, (key, value) in enumerate(additional_settings.items()):
            ttk.Label(frame2, text=key).grid(row=i + 1, column=0, sticky="W", padx=5, pady=5)
            ttk.Label(frame2, text=value).grid(row=i + 1, column=1, sticky="W", padx=5, pady=5)
            input_var = tk.StringVar()
            input_field = ttk.Entry(frame2, textvariable=input_var, width=30)
            input_field.grid(row=i + 1, column=2, padx=5, pady=5)
            input_fields.append((key, input_field))

        def on_submit():
            updates_server = {}
            updates_client = {}

            for key, field in input_fields:
                value = field.get().strip()
                if value:
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                    updates_server[key] = value
                    updates_client[key] = value

            update_yaml(config_server, updates_server)
            update_yaml(config_client, updates_client)

            print("Additional settings saved.")
            root2.destroy()

        ttk.Button(frame2, text="Submit", command=on_submit).grid(
            row=len(additional_settings) + 1, column=0, columnspan=3, pady=10
        )

    ttk.Button(frm, text="Additional Settings", command=open_additional_settings).grid(column=1, row=11)
    ttk.Button(frm, text="Start Training", command=lambda: confirm_start()).grid(column=1, row=12)


# --- Confirm Start ---
def confirm_start():
    start_choice = messagebox.askyesno("Confirm Start", "Setup ready. Do you want to start the training?")
    if start_choice:
        root.destroy()
        print("Configuration updated successfully for all three files.")


# --- Start GUI ---
show_task_options()
root.mainloop()

