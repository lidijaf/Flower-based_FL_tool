import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True

base_dir = os.path.dirname(__file__)
conf_dir = os.path.join(base_dir, "conf")
config_common = os.path.join(conf_dir, "config_common.yaml")
config_client = os.path.join(conf_dir, "config_client.yaml")
config_server = os.path.join(conf_dir, "config_server.yaml")

SUPPORTED_COMBINATIONS = {
    "classification": {
        "mnist": ["CNN_MNIST"],
        "custom dataset": ["CNN_MNIST"],
    },
    "anomaly detection": {
        "psm": ["Transformer", "Autoencoder"],
        "metro": ["Transformer", "Autoencoder"],
        "custom dataset": ["Transformer", "Autoencoder"],
    },
}

ALGORITHMS_BY_MODEL = {
    "CNN_MNIST": ["fedavg"],
    "Transformer": ["fedavg", "pfedme"],
    "Autoencoder": ["fedavg", "pfedme"],
}

DEFAULTS_COMMON = {
    "task": "classification",
    "dataset": "mnist",
    "model": "CNN_MNIST",
    "algorithm": "fedavg",
    "iid_partition": True,
    "input_c": 25,
    "output_c": 25,
    "win_size": 100,
    "k": 3,
    "step": 100,
    "seed": 42,
    "val_ratio": 0.2,
    "num_workers": 0,
    "num_labels_per_partition": 5,
    "min_data_per_partition": 10,
    "mean": 0.0,
    "sigma": 2.0,
    "device": "cpu",
    "data_path": "",
    "client_mode": "simulation",
    "node_role": "simulation",
    "server_address": "127.0.0.1:8080",
}

DEFAULTS_SERVER = {
    "num_rounds": 5,
    "num_clients": 2,
    "min_available_clients": 2,
    "min_fit_clients": 2,
    "min_evaluate_clients": 2,
}

DEFAULTS_CLIENT = {
    "epochs": 1,
    "batch_size": 32,
}

INT_FIELDS_SERVER = {
    "num_rounds",
    "num_clients",
    "min_available_clients",
    "min_fit_clients",
    "min_evaluate_clients",
}

INT_FIELDS_CLIENT = {
    "epochs",
    "batch_size",
}

FLOAT_FIELDS_COMMON = {
    "val_ratio",
    "mean",
    "sigma",
}

INT_FIELDS_COMMON = {
    "input_c",
    "output_c",
    "win_size",
    "k",
    "step",
    "seed",
    "num_workers",
    "num_labels_per_partition",
    "min_data_per_partition",
}

BOOL_FIELDS_COMMON = {
    "iid_partition",
}

DEFAULT_DATA_PATHS = {
    "mnist": "./data/mnist",
    "psm": "./data/Pooled Server Metrics (PSM)",
    "metro": "./data/metro",
}


def load_yaml_file(filepath: str) -> dict:
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r") as f:
        return yaml.load(f) or {}


def update_yaml(filepath: str, updates: dict) -> None:
    config = load_yaml_file(filepath)
    config.update(updates)
    with open(filepath, "w") as f:
        yaml.dump(config, f)


def cast_value(key: str, value: str):
    if key in INT_FIELDS_SERVER or key in INT_FIELDS_CLIENT or key in INT_FIELDS_COMMON:
        return int(value)
    if key in FLOAT_FIELDS_COMMON:
        return float(value)
    if key in BOOL_FIELDS_COMMON:
        return value.lower() in {"true", "1", "yes", "y"}
    return value


def validate_config(cfg: dict) -> tuple[bool, str]:
    task = cfg.get("task")
    dataset = cfg.get("dataset")
    model = cfg.get("model")
    algorithm = cfg.get("algorithm")
    data_path = (cfg.get("data_path") or "").strip()
    client_mode = cfg.get("client_mode")
    node_role = cfg.get("node_role")
    server_address = (cfg.get("server_address") or "").strip()

    if client_mode not in {"simulation", "real"}:
        return False, "Please choose simulation or real mode."

    if client_mode == "real" and node_role not in {"server", "client"}:
        return False, "In real mode, please choose whether this is a server or client machine."

    if client_mode == "simulation":
        node_role = "simulation"

    if task not in SUPPORTED_COMBINATIONS:
        return False, f"Unsupported task: {task}"

    if dataset not in SUPPORTED_COMBINATIONS[task]:
        return False, f"Dataset '{dataset}' is not supported for task '{task}'."

    if model not in SUPPORTED_COMBINATIONS[task][dataset]:
        return False, f"Model '{model}' is not supported for task '{task}' and dataset '{dataset}'."

    allowed_algorithms = ALGORITHMS_BY_MODEL.get(model, [])
    if algorithm not in allowed_algorithms:
        return False, f"Algorithm '{algorithm}' is not supported for model '{model}'."

    if not cfg.get("device"):
        return False, "Device must be set."

    if client_mode == "real" and node_role == "client":
        if not server_address:
            return False, "Real client mode requires a server address."
        if not data_path:
            return False, "Real client mode requires a local data path."
        if not os.path.exists(data_path):
            return False, f"Selected local data path does not exist:\n{data_path}"
        return True, "Configuration is valid for a real client machine."

    if client_mode == "real" and node_role == "server":
        if not server_address:
            return False, "Real server mode requires a server address."
        if data_path and not os.path.exists(data_path):
            return False, f"Selected server-side data path does not exist:\n{data_path}"
        return True, "Configuration is valid for a real server machine."

    # simulation mode
    if dataset == "custom dataset":
        if not data_path:
            return False, "Please choose a root directory for the custom simulation dataset."
        if not os.path.exists(data_path):
            return False, f"Selected data path does not exist:\n{data_path}"

        client1_dir = os.path.join(data_path, "client1")
        if not os.path.isdir(client1_dir):
            return False, (
                "Custom simulation datasets must be pre-partitioned.\n"
                "Expected at least a folder named 'client1' inside the selected root."
            )

        return True, "Custom simulation dataset root looks acceptable."

    # Built-in simulation datasets may use framework defaults or explicit paths if provided.
    if data_path and not os.path.exists(data_path):
        return False, f"Selected data path does not exist:\n{data_path}"

    return True, "Configuration is valid."


class FLConfigGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Federated Learning Setup")
        self.root.geometry("1260x920")
        self.root.minsize(1120, 820)

        self.common_cfg = {**DEFAULTS_COMMON, **load_yaml_file(config_common)}
        self.server_cfg = {**DEFAULTS_SERVER, **load_yaml_file(config_server)}
        self.client_cfg = {**DEFAULTS_CLIENT, **load_yaml_file(config_client)}

        self.client_mode_var = tk.StringVar(value=self.common_cfg.get("client_mode", "simulation"))
        self.node_role_var = tk.StringVar(value=self.common_cfg.get("node_role", "simulation"))

        self.task_var = tk.StringVar(value=self.common_cfg.get("task", "classification"))
        self.dataset_var = tk.StringVar(value=self.common_cfg.get("dataset", "mnist"))
        self.model_var = tk.StringVar(value=self.common_cfg.get("model", "CNN_MNIST"))
        self.algorithm_var = tk.StringVar(value=self.common_cfg.get("algorithm", "fedavg"))
        self.device_var = tk.StringVar(value=self.common_cfg.get("device", "cpu"))
        self.data_path_var = tk.StringVar(value=self.common_cfg.get("data_path", ""))
        self.server_address_var = tk.StringVar(value=self.common_cfg.get("server_address", "127.0.0.1:8080"))

        self.iid_var = tk.BooleanVar(value=bool(self.common_cfg.get("iid_partition", True)))

        self.status_var = tk.StringVar(value="Choose settings to begin.")
        self.summary_var = tk.StringVar(value="")
        self.help_var = tk.StringVar(value="")

        self.server_entries = {}
        self.client_entries = {}
        self.common_entries = {}

        self._build_ui()
        self._refresh_dataset_options()
        self._refresh_model_options()
        self._refresh_algorithm_options()
        self._apply_mode_visibility()
        self._refresh_summary()

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Section.TLabelframe.Label", font=("TkDefaultFont", 11, "bold"))

        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(container, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.main = ttk.Frame(self.canvas, padding=16)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.main, anchor="nw")

        self.main.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.main.columnconfigure(0, weight=3)
        self.main.columnconfigure(1, weight=2)
        self.main.rowconfigure(0, weight=1)

        left = ttk.Frame(self.main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        right = ttk.Frame(self.main)
        right.grid(row=0, column=1, sticky="nsew")

        self._build_mode_section(left)
        self._build_experiment_section(left)
        self._build_common_section(left)
        self._build_training_section(left)
        self._build_summary_section(right)
        self._build_action_section(right)

    def _on_frame_configure(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _build_mode_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Execution Mode", style="Section.TLabelframe", padding=12)
        frame.pack(fill="x", pady=(0, 10))

        for i in range(2):
            frame.columnconfigure(i, weight=1)

        ttk.Label(frame, text="Mode").grid(row=0, column=0, sticky="w", pady=4)
        mode_combo = ttk.Combobox(
            frame,
            textvariable=self.client_mode_var,
            state="readonly",
            values=["simulation", "real"],
        )
        mode_combo.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        mode_combo.bind("<<ComboboxSelected>>", self._on_mode_changed)

        ttk.Label(frame, text="Configure").grid(row=0, column=1, sticky="w", pady=4)
        self.node_role_combo = ttk.Combobox(
            frame,
            textvariable=self.node_role_var,
            state="readonly",
            values=["simulation"],
        )
        self.node_role_combo.grid(row=1, column=1, sticky="ew")
        self.node_role_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_mode_visibility())

    def _build_experiment_section(self, parent):
        self.experiment_frame = ttk.LabelFrame(parent, text="Experiment", style="Section.TLabelframe", padding=12)
        self.experiment_frame.pack(fill="x", pady=(0, 10))

        for i in range(2):
            self.experiment_frame.columnconfigure(i, weight=1)

        ttk.Label(self.experiment_frame, text="Task").grid(row=0, column=0, sticky="w", pady=4)
        task_combo = ttk.Combobox(
            self.experiment_frame,
            textvariable=self.task_var,
            state="readonly",
            values=list(SUPPORTED_COMBINATIONS.keys()),
        )
        task_combo.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        task_combo.bind("<<ComboboxSelected>>", self._on_task_changed)

        ttk.Label(self.experiment_frame, text="Dataset").grid(row=0, column=1, sticky="w", pady=4)
        self.dataset_combo = ttk.Combobox(
            self.experiment_frame,
            textvariable=self.dataset_var,
            state="readonly",
        )
        self.dataset_combo.grid(row=1, column=1, sticky="ew")
        self.dataset_combo.bind("<<ComboboxSelected>>", self._on_dataset_changed)

        ttk.Label(self.experiment_frame, text="Model").grid(row=2, column=0, sticky="w", pady=(12, 4))
        self.model_combo = ttk.Combobox(
            self.experiment_frame,
            textvariable=self.model_var,
            state="readonly",
        )
        self.model_combo.grid(row=3, column=0, sticky="ew", padx=(0, 8))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_changed)

        ttk.Label(self.experiment_frame, text="Algorithm").grid(row=2, column=1, sticky="w", pady=(12, 4))
        self.algorithm_combo = ttk.Combobox(
            self.experiment_frame,
            textvariable=self.algorithm_var,
            state="readonly",
        )
        self.algorithm_combo.grid(row=3, column=1, sticky="ew")
        self.algorithm_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_summary())

        ttk.Label(self.experiment_frame, text="Device").grid(row=4, column=0, sticky="w", pady=(12, 4))
        device_combo = ttk.Combobox(
            self.experiment_frame,
            textvariable=self.device_var,
            state="readonly",
            values=["cpu", "cuda"],
        )
        device_combo.grid(row=5, column=0, sticky="ew", padx=(0, 8))
        device_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_summary())

        ttk.Label(self.experiment_frame, text="Server address").grid(row=4, column=1, sticky="w", pady=(12, 4))
        self.server_address_entry = ttk.Entry(self.experiment_frame, textvariable=self.server_address_var)
        self.server_address_entry.grid(row=5, column=1, sticky="ew")
        self.server_address_entry.bind("<KeyRelease>", lambda e: self._refresh_summary())

        ttk.Label(self.experiment_frame, text="Data path").grid(row=6, column=0, sticky="w", pady=(12, 4))
        path_frame = ttk.Frame(self.experiment_frame)
        path_frame.grid(row=7, column=0, columnspan=2, sticky="ew")
        path_frame.columnconfigure(0, weight=1)

        self.data_path_entry = ttk.Entry(path_frame, textvariable=self.data_path_var)
        self.data_path_entry.grid(row=0, column=0, sticky="ew")
        self.data_path_entry.bind("<KeyRelease>", lambda e: self._refresh_summary())

        self.browse_btn = ttk.Button(path_frame, text="Browse...", command=self._browse_data_path)
        self.browse_btn.grid(row=0, column=1, padx=(6, 0))

        self.help_label = ttk.Label(
            self.experiment_frame,
            textvariable=self.help_var,
            wraplength=760,
            justify="left",
        )
        self.help_label.grid(row=8, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def _build_common_section(self, parent):
        self.common_frame = ttk.LabelFrame(parent, text="Dataset / Common Settings", style="Section.TLabelframe", padding=12)
        self.common_frame.pack(fill="x", pady=(0, 10))

        fields = [
            ("input_c", self.common_cfg.get("input_c")),
            ("output_c", self.common_cfg.get("output_c")),
            ("win_size", self.common_cfg.get("win_size")),
            ("k", self.common_cfg.get("k")),
            ("step", self.common_cfg.get("step")),
            ("seed", self.common_cfg.get("seed")),
            ("val_ratio", self.common_cfg.get("val_ratio")),
            ("num_workers", self.common_cfg.get("num_workers")),
            ("num_labels_per_partition", self.common_cfg.get("num_labels_per_partition")),
            ("min_data_per_partition", self.common_cfg.get("min_data_per_partition")),
            ("mean", self.common_cfg.get("mean")),
            ("sigma", self.common_cfg.get("sigma")),
        ]

        for idx, (key, value) in enumerate(fields):
            row = idx // 2
            col = (idx % 2) * 2
            ttk.Label(self.common_frame, text=key).grid(row=row, column=col, sticky="w", padx=(0, 8), pady=4)
            entry = ttk.Entry(self.common_frame)
            entry.insert(0, str(value))
            entry.grid(row=row, column=col + 1, sticky="ew", padx=(0, 16), pady=4)
            entry.bind("<KeyRelease>", lambda e: self._refresh_summary())
            self.common_entries[key] = entry

        total_rows = (len(fields) + 1) // 2
        ttk.Checkbutton(
            self.common_frame,
            text="IID partition",
            variable=self.iid_var,
            command=self._refresh_summary,
        ).grid(row=total_rows + 1, column=0, sticky="w", pady=(8, 0))

        for c in range(4):
            self.common_frame.columnconfigure(c, weight=1)

    def _build_training_section(self, parent):
        self.training_frame = ttk.LabelFrame(parent, text="Server / Client Settings", style="Section.TLabelframe", padding=12)
        self.training_frame.pack(fill="x", pady=(0, 10))

        server_fields = [
            ("num_rounds", self.server_cfg.get("num_rounds")),
            ("num_clients", self.server_cfg.get("num_clients")),
            ("min_available_clients", self.server_cfg.get("min_available_clients")),
            ("min_fit_clients", self.server_cfg.get("min_fit_clients")),
            ("min_evaluate_clients", self.server_cfg.get("min_evaluate_clients")),
        ]

        client_fields = [
            ("epochs", self.client_cfg.get("epochs")),
            ("batch_size", self.client_cfg.get("batch_size")),
        ]

        self.server_title = ttk.Label(self.training_frame, text="Server", font=("TkDefaultFont", 10, "bold"))
        self.server_title.grid(row=0, column=0, sticky="w")
        self.client_title = ttk.Label(self.training_frame, text="Client", font=("TkDefaultFont", 10, "bold"))
        self.client_title.grid(row=0, column=2, sticky="w")

        for idx, (key, value) in enumerate(server_fields, start=1):
            ttk.Label(self.training_frame, text=key).grid(row=idx, column=0, sticky="w", pady=4, padx=(0, 8))
            entry = ttk.Entry(self.training_frame)
            entry.insert(0, str(value))
            entry.grid(row=idx, column=1, sticky="ew", padx=(0, 16), pady=4)
            entry.bind("<KeyRelease>", lambda e: self._refresh_summary())
            self.server_entries[key] = entry

        for idx, (key, value) in enumerate(client_fields, start=1):
            ttk.Label(self.training_frame, text=key).grid(row=idx, column=2, sticky="w", pady=4, padx=(0, 8))
            entry = ttk.Entry(self.training_frame)
            entry.insert(0, str(value))
            entry.grid(row=idx, column=3, sticky="ew", pady=4)
            entry.bind("<KeyRelease>", lambda e: self._refresh_summary())
            self.client_entries[key] = entry

        for c in range(4):
            self.training_frame.columnconfigure(c, weight=1)

    def _build_summary_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Summary", style="Section.TLabelframe", padding=12)
        frame.pack(fill="both", expand=True, pady=(0, 10))

        ttk.Label(
            frame,
            textvariable=self.status_var,
            foreground="#1f5f3b",
            wraplength=340,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))

        ttk.Label(
            frame,
            textvariable=self.summary_var,
            justify="left",
            wraplength=360,
        ).pack(anchor="nw", fill="both", expand=True)

    def _build_action_section(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill="x")

        ttk.Button(frame, text="Validate", command=self._validate_only).pack(side="left")
        ttk.Button(frame, text="Save Config", command=self._save_config).pack(side="left", padx=8)
        ttk.Button(frame, text="Close", command=self.root.destroy).pack(side="right")

    def _refresh_dataset_options(self):
        task = self.task_var.get()
        datasets = list(SUPPORTED_COMBINATIONS.get(task, {}).keys())
        self.dataset_combo["values"] = datasets
        if self.dataset_var.get() not in datasets and datasets:
            self.dataset_var.set(datasets[0])

    def _refresh_model_options(self):
        task = self.task_var.get()
        dataset = self.dataset_var.get()
        models = SUPPORTED_COMBINATIONS.get(task, {}).get(dataset, [])
        self.model_combo["values"] = models
        if self.model_var.get() not in models and models:
            self.model_var.set(models[0])
        self.model_combo.configure(state="readonly")

    def _refresh_algorithm_options(self):
        model = self.model_var.get()
        algorithms = ALGORITHMS_BY_MODEL.get(model, ["fedavg"])
        self.algorithm_combo["values"] = algorithms
        if self.algorithm_var.get() not in algorithms and algorithms:
            self.algorithm_var.set(algorithms[0])
        self.algorithm_combo.configure(state="readonly")

    def _on_mode_changed(self, _event=None):
        if self.client_mode_var.get() == "simulation":
            self.node_role_combo["values"] = ["simulation"]
            self.node_role_var.set("simulation")
        else:
            self.node_role_combo["values"] = ["server", "client"]
            if self.node_role_var.get() not in {"server", "client"}:
                self.node_role_var.set("server")
        self._apply_mode_visibility()

    def _on_task_changed(self, _event=None):
        self._refresh_dataset_options()
        self._refresh_model_options()
        self._refresh_algorithm_options()
        self._refresh_summary()

    def _on_dataset_changed(self, _event=None):
        self._apply_mode_visibility()

    def _on_model_changed(self, _event=None):
        self._refresh_algorithm_options()
        self._refresh_summary()

    def _browse_data_path(self):
        selected = filedialog.askdirectory(title="Select dataset folder", initialdir=base_dir)
        if selected:
            self.data_path_var.set(selected)
            self._refresh_summary()

    def _apply_mode_visibility(self):
        mode = self.client_mode_var.get()
        role = self.node_role_var.get()
        dataset = self.dataset_var.get()

        is_simulation = mode == "simulation"
        is_real_server = mode == "real" and role == "server"
        is_real_client = mode == "real" and role == "client"
        is_custom_dataset = dataset == "custom dataset"

        # server address
        if is_simulation:
            self.server_address_entry.configure(state="disabled")
        else:
            self.server_address_entry.configure(state="normal")

        # data path behavior
        if is_real_client:
            self.data_path_entry.configure(state="normal")
            self.browse_btn.configure(state="normal")
        elif is_real_server:
            self.data_path_entry.configure(state="normal")
            self.browse_btn.configure(state="normal")
        elif is_simulation and is_custom_dataset:
            self.data_path_entry.configure(state="normal")
            self.browse_btn.configure(state="normal")
        else:
            self.data_path_entry.configure(state="disabled")
            self.browse_btn.configure(state="disabled")

            dataset = self.dataset_var.get()
            default_path = DEFAULT_DATA_PATHS.get(dataset, "")
            self.data_path_var.set(default_path)

        # labels
        if is_real_server:
            self.server_title.configure(text="Server")
            self.client_title.configure(text="Client (saved but not required on this machine)")
        elif is_real_client:
            self.server_title.configure(text="Server (saved but not required on this machine)")
            self.client_title.configure(text="Client")
        else:
            self.server_title.configure(text="Server")
            self.client_title.configure(text="Client")

        # help text
        if is_simulation and is_custom_dataset:
            if self.task_var.get() == "anomaly detection":
                self.help_var.set(
                    "Simulation + custom dataset:\n"
                    "Select the dataset root directory. It should contain folders per client, "
                    "for example client1, client2, ...\n"
                    "For anomaly detection, each client folder should contain files like:\n"
                    "train.csv, test.csv, test_label.csv"
                )
            else:
                self.help_var.set(
                    "Simulation + custom dataset:\n"
                    "Select the dataset root directory. It should contain folders per client, "
                    "for example client1, client2, ...\n"
                    "For classification, each client folder should contain files like:\n"
                    "train.pt, test.pt"
                )
        elif is_real_client:
            self.help_var.set(
                "Real client mode:\n"
                "Select the local data directory for this client machine. "
                "This path will be used as-is, without adding client1/client2 suffixes."
            )
        elif is_real_server:
            self.help_var.set(
                "Real server mode:\n"
                "Configure the server address and server-side FL settings.\n"
                "If your server performs evaluation or threshold-related operations, "
                "you may optionally provide a server-side data path here."
            )
        else:
            self.help_var.set(
                "Simulation mode:\n"
                "Built-in datasets use the framework's default dataset roots.\n"
                "The selected dataset root will be written to config automatically.\n"
                "Choose 'custom dataset' if you want to provide your own simulation dataset root."
            )

        self._refresh_dataset_options()
        self._refresh_model_options()
        self._refresh_algorithm_options()
        self._refresh_summary()

    def _collect_config(self) -> dict:
        cfg = {
            "client_mode": self.client_mode_var.get(),
            "node_role": self.node_role_var.get(),
            "task": self.task_var.get(),
            "dataset": self.dataset_var.get(),
            "model": self.model_var.get(),
            "algorithm": self.algorithm_var.get(),
            "device": self.device_var.get(),
            "data_path": self.data_path_var.get().strip(),
            "iid_partition": bool(self.iid_var.get()),
            "server_address": self.server_address_var.get().strip(),
        }

        for key, entry in self.common_entries.items():
            raw = entry.get().strip()
            if raw:
                cfg[key] = cast_value(key, raw)

        return cfg

    def _collect_server_config(self) -> dict:
        cfg = {}
        for key, entry in self.server_entries.items():
            raw = entry.get().strip()
            if raw:
                cfg[key] = cast_value(key, raw)
        return cfg

    def _collect_client_config(self) -> dict:
        cfg = {}
        for key, entry in self.client_entries.items():
            raw = entry.get().strip()
            if raw:
                cfg[key] = cast_value(key, raw)
        return cfg

    def _refresh_summary(self):
        try:
            common_cfg = self._collect_config()
            valid, msg = validate_config(common_cfg)
            self.status_var.set(("✅ " if valid else "⚠️ ") + msg)
        except Exception as exc:
            self.status_var.set(f"⚠️ Invalid values: {exc}")

        summary = [
            f"Mode: {self.client_mode_var.get()}",
            f"Role: {self.node_role_var.get()}",
            f"Task: {self.task_var.get()}",
            f"Dataset: {self.dataset_var.get()}",
            f"Model: {self.model_var.get()}",
            f"Algorithm: {self.algorithm_var.get()}",
            f"Device: {self.device_var.get()}",
            f"Server address: {self.server_address_var.get() or '(not set)'}",
            f"Data path: {self.data_path_var.get() or '(not set)'}",
            "",
            "Files:",
            f"- common: {config_common}",
            f"- client: {config_client}",
            f"- server: {config_server}",
        ]
        self.summary_var.set("\n".join(summary))

    def _validate_only(self):
        try:
            common_cfg = self._collect_config()
            valid, msg = validate_config(common_cfg)
            if valid:
                messagebox.showinfo("Validation", msg)
            else:
                messagebox.showerror("Validation failed", msg)
            self._refresh_summary()
        except Exception as exc:
            messagebox.showerror("Validation error", str(exc))

    def _save_config(self):
        try:
            common_cfg = self._collect_config()
            server_cfg = self._collect_server_config()
            client_cfg = self._collect_client_config()

            valid, msg = validate_config(common_cfg)
            if not valid:
                messagebox.showerror("Cannot save config", msg)
                self._refresh_summary()
                return

            update_yaml(config_common, common_cfg)

            mode = common_cfg["client_mode"]
            role = common_cfg["node_role"]

            if mode == "simulation":
                update_yaml(config_server, server_cfg)
                update_yaml(config_client, client_cfg)
            elif mode == "real" and role == "server":
                update_yaml(config_server, server_cfg)
            elif mode == "real" and role == "client":
                update_yaml(config_client, client_cfg)

            self._refresh_summary()
            messagebox.showinfo("Saved", "Configuration saved successfully.")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

if __name__ == "__main__":
    root = tk.Tk()
    app = FLConfigGUI(root)
    root.mainloop()
