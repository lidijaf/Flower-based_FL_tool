# Flower-based Federated Learning Tool

A modular framework for running **Federated Learning (FL)** experiments using Flower.

This tool supports both **simulation environments** and **real-world distributed setups**, with a user-friendly GUI for configuration and a flexible architecture for research and experimentation.

---

## 🚀 Key Features

- 🖥️ Interactive GUI Configuration  
  - No manual YAML editing required  
  - Valid combinations only (task–dataset–model–algorithm)

- 🔄 Two Operating Modes  
  - Simulation Mode → centralized experiments  
  - Real Mode → distributed client/server setup  

- 🧠 Supported Tasks  
  - Classification (e.g., MNIST)  
  - Anomaly Detection (Autoencoder, Transformer)  
  
  - 🔬 Supported Algorithms  
  - FedAvg  
  - pFedMe  
  - pFedMeNew  
  - DRFL (prototype, currently CNN-only)  

- 🧩 Modular Design  
  - Plug-and-play models, datasets, and strategies  
  - Easy to extend for research purposes  

- 📂 Custom Dataset Support  
  - Works with your own data (CSV or tensor formats)  

- ⚙️ Unified Client Launcher  
  - No need to manually pass dataset paths at runtime  

---

## ⚙️ Installation

### Requirements
- Python 3.10 or 3.11  

### Setup

```bash
git clone <your-repo-url>
cd Flower-based_FL_tool

python -m venv flower_env
source flower_env/bin/activate      # Linux/macOS
# or
flower_env\Scripts\activate         # Windows

pip install -r requirements_stable.txt
```

---

## 🧭 Configuration

Launch the configuration interface:

```bash
python configure.py
```

---

## 🧠 Configuration Modes

### 🔹 1. Simulation Mode

Used for experiments on a single machine.

- Data is split into multiple virtual clients  
- Configure:
  - task, dataset, model  
  - number of clients  
  - FL parameters  

---

### 🔹 2. Real Mode

Used for distributed systems.

You choose:

#### 🖥️ Server Setup
- Number of rounds  
- Aggregation parameters  
- Global model settings  

#### 💻 Client Setup
- Local dataset path  
- Server address  

Each client runs independently using its own configuration.

---

## 📁 Custom Dataset Support

### Simulation Mode

Your dataset must follow:

```
your_dataset/
├── client1/
│   ├── train.csv
│   ├── test.csv
│   └── val.csv
├── client2/
│   ├── ...
```

Each folder represents one simulated client.

---

### Real Mode

Each client simply points to its local dataset directory.

---
## 🧹 Preprocessing and Dataset Preparation

The framework includes a **modular preprocessing and dataset preparation module** under:

```
data_preprocessing/
```

This module provides a unified way to:
- download built-in datasets  
- preprocess structured and tabular data  
- generate federated client splits  
- inject anomalies for anomaly detection  
- export data in CSV or PyTorch formats  

---

### ⚙️ Main Entry Point

Datasets are prepared using:

```bash
python use_dataset.py <dataset_name> [options]
```

---

### 📦 Supported Datasets

| Dataset Type | Name | Description |
|-------------|------|------------|
| Built-in | `mnist`, `cifar10`, `fmnist` | Automatically downloaded |
| Structured | `act` | Event-based JSON data |
| Tabular (Anomaly) | `metropt` | Industrial anomaly detection dataset |
| Time-series | `psm` | Pooled Server Metrics dataset |
| Generic | `tabular` | Any CSV dataset |

---

### ▶️ Examples

#### 📥 Built-in datasets

```bash
python use_dataset.py mnist
python use_dataset.py cifar10 --output_dir data/cifar10
```

---

#### 🧠 ACT dataset (JSON → FL-ready)

```bash
# Standard preprocessing
python use_dataset.py act --input_path data/raw/run1.json

# Balanced client partitioning
python use_dataset.py act \
    --input_path data/raw/run1.json \
    --mode balanced \
    --num_clients 4

# Anomaly injection
python use_dataset.py act \
    --input_path data/raw/run1.json \
    --mode anomaly
```

---

#### 🏭 MetroPT dataset

```bash
python use_dataset.py metropt \
    --input_path data/raw/metro.csv \
    --num_clients 2 \
    --output_dir data/metropt
```

---

#### 📊 PSM dataset

```bash
python use_dataset.py psm \
    --input_path "data/Pooled Server Metrics (PSM)" \
    --num_clients 2 \
    --output_dir data/psm
```

Expected structure:

```
train.csv
test.csv
test_label.csv
```

---

#### 📄 Generic tabular dataset

```bash
python use_dataset.py tabular \
    --input_path data/raw/sample.csv \
    --num_clients 3
```

---

### 📁 Output Structure

Prepared datasets are exported into client-specific folders:

```
client1/
client2/
client3/
...
```

Each client contains:

#### CSV format
```
train.csv
val.csv
test.csv
```

#### or PyTorch format
```
train.pt
val.pt
test.pt
```

---

### 🧠 Preprocessing Capabilities

The preprocessing module supports:

- JSON flattening (ACT)
- feature engineering (e.g. temporal features)
- encoding and normalization
- flexible client partitioning
- balanced client creation
- train / validation / test splitting
- anomaly injection
- anomaly-aware data splitting
- CSV and tensor export

---

### 🔄 Workflow Integration

Dataset preparation is intentionally **decoupled from training**.

Recommended workflow:

1. Prepare dataset:
```bash
python use_dataset.py ...
```

2. Configure experiment:
```bash
python configure.py
```

3. Run training:
```bash
python start_server.py
```

---

### ⚠️ Note on Data Modules

- `data_preprocessing/` → full preprocessing framework  
- `data_preparations/` → lightweight compatibility layer used by existing clients  

All new preprocessing logic should be added to `data_preprocessing/`.

## ▶️ Running the System

### 1. Start the Server

```bash
python start_server.py
```

---

### 2. Start Clients

```bash
python -m clients.start_client --client_id 1
```

- No need to pass dataset path  
- Everything is read from configuration  

---

## 🧪 Example Workflow

1. Run GUI:
```bash
python configure.py
```

2. Choose:
- Simulation mode  
- MNIST + CNN  
- FedAvg  

3. Start server:
```bash
python start_server.py
```

4. Start clients:
```bash
python -m clients.start_client --client_id 1
python -m clients.start_client --client_id 2
```

---

## 📊 Outputs

All results are stored in:

```
outputs/
```

Includes:
- Training curves (.png)  
- Metrics (.npy)  
- Anomaly outputs (.npz)  

---

## 🏗️ Project Structure

```
.
├── algorithms/
├── clients/
├── servers/
│   └── strategies/
├── models/
├── data_preparations/
├── conf/
├── data/
├── utils/
├── outputs/
│
├── configure.py
├── start_server.py
```

---

## 🧩 Extending the Framework

### Add a new model
- Add it in `models/`  
- Register it in GUI  

### Add a new dataset
- Add loader in `data_preparations/`  
- Define structure in GUI validation  

### Add a new algorithm
- Implement client-side logic in:
  - `algorithms/`
- If needed, implement custom server aggregation in:
  - `servers/strategies/`
---

---

## 🧪 Experimental Algorithms

### DRFL (Distributionally Robust Federated Learning)

DRFL is implemented as an experimental client-server pipeline for CNN-based classification tasks.

- Clients transmit serialized gradients to the server  
- The server clusters clients based on gradient similarity  
- The cluster with the highest average loss is selected  
- A normalized robust gradient update is applied to the global model  

With a small number of clients, clustering may isolate individual clients.  
In such cases, the system falls back to a single shared cluster to ensure stable training.

This implementation is intended for research and experimentation, and is most effective in settings with multiple heterogeneous clients.
## 🧑‍💻 Running on a Cluster (SLURM)

Example:

```bash
python slurm-launch.py   --exp-name flower_server   --num-nodes 1   --partition all   --command "python start_server.py"
```

Run clients:

```bash
bash launch_all_clients.sh
```

---

## ⚠️ Important Notes

- Configuration is stored in:
  - `config_common.yaml`  
  - `config_server.yaml`  
  - `config_client.yaml`  

- Recommended `.gitignore` entries:
  outputs/
  __pycache__/
  *.pth
  *.npz

---

## 📜 License

This project is licensed under the MIT License.

---

## ⚠️ Disclaimer

This tool was developed within the TaRDIS project  
(Grant agreement No. 101093006), funded by the Swiss State Secretariat for Education, Research and Innovation (SERI).

---

## 📧 Contact

- nemanjab4h@gmail.com  
- lidija.fodor@dmi.uns.ac.rs  

---

## ⭐ Final Note

This tool is designed for:
- research reproducibility  
- rapid FL experimentation  
- bridging simulation and real deployment  
