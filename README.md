# Flower-based Federated Learning Tool

A modular framework for running **Federated Learning (FL)** experiments using Flower.

This framework supports both **simulation environments** and **real-world distributed deployments**, with a user-friendly GUI for configuration and a flexible architecture for research and experimentation.

---

## 🚀 Key Features

-  Interactive GUI Configuration 
  - No manual YAML editing required
  - Valid combinations only (task–dataset–model–algorithm)

- Two Operating Modes
  - Simulation Mode → centralized experiments
  - Real Mode → distributed client/server setup

- Supported Tasks
  - Classification (e.g., MNIST)
  - Anomaly Detection (Autoencoder, Transformer)
  
-  Supported Algorithms
  - FedAvg
  - FedAvg+KD
  - pFedMe
  - pFedMeNew
  - DRFL (prototype, currently CNN-only)

- Modular Design
  - Plug-and-play models, datasets, and strategies
  - Easy to extend for research purposes

- Custom Dataset Support
  - Works with your own data (CSV or tensor formats)

- Unified Client Launcher
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

### Configuration Modes

#### 1. Simulation Mode

Used for experiments on a single machine.

- Data is split into multiple virtual clients
- Configure:
  - task, dataset, model
  - number of clients
  - FL parameters

---

#### 2. Real Mode

Used for distributed systems.

You choose:

##### Server Setup
- Number of rounds
- Aggregation parameters
- Global model settings

##### Client Setup
- Local dataset path
- Server address

Each client runs independently using its own configuration.

---

### Custom Dataset Support

#### Simulation Mode

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

#### Real Mode

Each client simply points to its local dataset directory.

---

## 📂 Dataset Acquisition

The framework supports both publicly available datasets and project-specific datasets.

### MNIST

MNIST is automatically downloaded through torchvision during dataset preparation.

Example:

```bash
python3 use_dataset.py mnist --num_clients 2
```

No manual download is required.

---

### Fashion-MNIST

Fashion-MNIST is also downloaded automatically through torchvision.

Example:

```bash
python3 use_dataset.py fmnist --num_clients 2
```

No manual download is required.

---

### CIFAR-10

CIFAR-10 can be prepared using:

```bash
python3 use_dataset.py cifar10 --num_clients 2
```

The dataset will be downloaded automatically if it is not already present.

---

### MetroPT-3

MetroPT is a public predictive-maintenance dataset originating from the Metro do Porto air-compressor system.

Users must first obtain the raw MetroPT dataset and place it in a local directory.

Dataset source:

- Scientific Data paper:
  https://www.nature.com/articles/s41597-022-01877-3

- UCI Machine Learning Repository:
  https://archive.ics.uci.edu/dataset/791/metropt+3+dataset

Example:

```bash
python3 use_dataset.py metropt     --input_dir raw_data/metropt     --output_dir data/metropt
```

For Transformer-based anomaly detection, windowed datasets can be generated using:

```bash
python3 data_preprocessing/scripts/prepare_windowed_dataset.py     --input_dir data/metropt     --output_dir data/metropt_transformer     --win_size 20     --step 5     --label_mode max
```

---

### PSM

PSM (Pooled Server Metrics) is a public anomaly-detection benchmark dataset, containing server telemetry collected from multiple eBay application servers.

Dataset sources:

- Original repository:
  https://github.com/eBay/RANSynCoders

- Reference publication:
  Abdulaal et al., "Practical Approach to Asynchronous Multivariate Time Series
  Anomaly Detection and Localization", KDD 2021.

Users should download the original PSM files and place them in a local directory containing:

```text
train.csv
test.csv
test_label.csv
```

Dataset preparation can then be performed using the provided preprocessing utilities.

---

### ACT Dataset

The ACT dataset used within the TaRDIS project is not distributed as part of this repository.

To use the ACT preprocessing workflow, users must provide:

- an input JSON dataset
- a dataset-specific schema/configuration file describing the JSON structure

The preprocessing pipeline is implemented in:

```text
data_preprocessing/
```

and can be adapted to other JSON-based industrial datasets.

---

### Supported Dataset Types

#### Classification

- MNIST
- Fashion-MNIST
- CIFAR-10

#### Anomaly Detection

- MetroPT-3
- PSM
- ACT (project-specific)

---

### Recommended Workflow

```text
Acquire dataset
        ↓
Run dataset preparation
        ↓
Generate client partitions
        ↓
(Optional) Generate Transformer windows
        ↓
Run federated training
        ↓
Evaluate / Infer
```
---

## 📁 Preprocessing and Dataset Preparation

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

### Main Entry Point

Datasets are prepared using:

```bash
python use_dataset.py <dataset_name> [options]
```

---

### Supported Datasets

| Dataset Type | Name | Description |
|-------------|------|------------|
| Built-in | `mnist`, `cifar10`, `fmnist` | Automatically downloaded |
| Structured | `act` | Event-based JSON data |
| Tabular (Anomaly) | `metropt` | Industrial anomaly detection dataset |
| Time-series | `psm` | Pooled Server Metrics dataset |
| Generic | `tabular` | Any CSV dataset |

---

### Examples

#### Built-in datasets

```bash
python use_dataset.py mnist
python use_dataset.py cifar10 --output_dir data/cifar10
```

---

#### ACT dataset (JSON → FL-ready)

The framework contains a reusable preprocessing recipe developed for the ACT use case within the TaRDIS project.

To protect project-specific data structures and metadata, ACT schemas and sample datasets are not distributed with the public repository.

Users who wish to use this recipe must provide:

-an input JSON dataset
-a dataset-specific schema/configuration file describing the structure of the JSON data

Example:

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

The ACT recipe demonstrates how the preprocessing framework can be applied to schema-driven JSON event streams while keeping dataset-specific details outside the public repository.

---

#### MetroPT dataset

```bash
python use_dataset.py metropt \
    --input_path data/raw/metro.csv \
    --num_clients 2 \
    --output_dir data/metropt
```

---

#### PSM dataset

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

#### Generic tabular dataset

```bash
python use_dataset.py tabular \
    --input_path data/raw/sample.csv \
    --num_clients 3
```

---

### Output Structure

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

### Preprocessing Capabilities

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

### Workflow Integration

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

### Note on Data Modules

- `data_preprocessing/` → full preprocessing framework
All preprocessing, dataset preparation, tensor export, and window generation logic is implemented in `data_preprocessing/`.


### Transformer Dataset Preparation

Transformer-based anomaly detection models expect windowed tensors with shape:

```text
[num_windows, win_size, num_features]
```

Autoencoder models use flat tensors:

```text
[num_samples, num_features]
```

To create Transformer-ready datasets from prepared client folders:

```bash
python3 -m data_preprocessing.scripts.prepare_windowed_dataset     --input_dir data/metropt     --output_dir data/metropt_transformer     --win_size 20     --step 5     --label_mode sequence
```

For MetroPT Transformer experiments:

```yaml
task: anomaly detection
dataset: metropt
model: Transformer
algorithm: fedavg
input_c: 14
output_c: 14
win_size: 20
step: 5
data_path: './data/metropt_transformer'
```

For MetroPT Autoencoder experiments:

```yaml
task: anomaly detection
dataset: metropt
model: Autoencoder
algorithm: fedavg
data_path: './data/metropt'
```

`label_mode: sequence` preserves one label per timestep inside each generated window and is required by the current Transformer evaluation pipeline.


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

## 📈 Monitoring and KPI Tracking

The framework includes built-in monitoring for federated learning experiments.

### Client-side Metrics
- Training time
- Evaluation time
- Memory usage (start, end, delta, peak)
- Communication volume (bytes and MB)
- Number of transmitted parameters

### Server-side Metrics
- Total training duration
- Aggregation time per round
- Aggregation memory usage
- Number of participating clients
- Round-level aggregation statistics

Monitoring logs are written to:

```text
outputs/monitoring/
```

Metrics are stored in JSONL format and can be post-processed to generate experiment summaries and KPI reports.

---

## 📡 Communication Compression

The framework supports optional FP16 parameter transmission to reduce communication overhead.

Configuration:

```yaml
transmission_precision: fp16
```

Supported values:

```yaml
transmission_precision: fp32
transmission_precision: fp16
```

Model training remains in FP32 while transmitted parameters are quantized before communication and restored to the model datatype after reception.

This feature was introduced to support lightweight FL deployments and communication-efficiency experiments.

---

## 🧠 Knowledge Distillation

Knowledge Distillation is available through:

```yaml
algorithm: fedavg+KD
```

Optional parameters:

```yaml
kd_temperature: 2.0
kd_alpha: 0.5
```

This variant extends FedAvg with a teacher-student distillation step and can be used as a lightweight learning technique for classification workloads.

---

## 💾 Global Model Persistence

The server can automatically save the final global model after federated training.

Configuration:

```yaml
save_global_model: true
global_model_output_path: outputs/models/final_global_model.pt
```

Saved checkpoints can later be used for:
- Warm-start training
- Incremental retraining
- Standalone inference
- External evaluation

---

## 🔄 Warm Start and Incremental Training

Training can resume from a previously saved global model.

Configuration:

```yaml
warm_start: true
warm_start_model_path: outputs/models/final_global_model.pt
```

When enabled, the server initializes the global model using the provided checkpoint instead of random initialization.


## Model Lifecycle

```text
Dataset Preparation
        ↓
Federated Training
        ↓
Save Global Model
        ↓
Warm Start / Inference / Evaluation
```

---


## 🔍 Inference

The framework supports standalone inference using previously trained global models.

Example:

```bash
python3 run_inference.py \
    --model_path outputs/models/final_global_model.pt \
    --input_path data/mnist/client1/test.pt \
    --output_dir outputs/inference/cnn_mnist_test
```

Generated outputs may include:

```text
predictions.npy
probabilities.npy
scores.npy
labels.npy
inference_report.json
```

Inference utilities support both classification and anomaly-detection workflows.

---

## 🧪 Evaluation Utilities

The framework includes reusable evaluation components:

```text
evaluation/
├── evaluator.py
├── inference.py
├── results.py
```

These modules provide:
- Metric computation
- Prediction handling
- Result export
- Inference support
- Anomaly-detection evaluation



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
- Monitoring logs (.jsonl)
- Saved global models: outputs/models/
- Inference outputs

Typical structure:

```text
outputs/
├── inference/
├── mislabeled_experiments/
├── models/
├── monitoring/
```
  
---

## ▶️ Validation and Recommended Test Configurations

The following configurations are recommended for validating a fresh installation.

### CNN + FedAvg

```yaml
task: classification
dataset: mnist
model: CNN_MNIST
algorithm: fedavg
num_rounds: 2
```

Expected outcome:

- training completes successfully
- monitoring logs are generated
- global model can be saved
- inference can be executed on the saved model

---

### CNN + FedAvg+KD

```yaml
task: classification
dataset: mnist
model: CNN_MNIST
algorithm: fedavg+KD
num_rounds: 2
```

Expected outcome:

- training completes successfully
- knowledge distillation loss is applied
- monitoring logs are generated

---

### CNN + DRFL

```yaml
task: classification
dataset: mnist
model: CNN_MNIST
algorithm: drfl
num_rounds: 2
```

Expected outcome:

- training completes successfully
- DRFL aggregation executes correctly

---

### Autoencoder + FedAvg

```yaml
task: anomaly detection
dataset: metro
model: Autoencoder
algorithm: fedavg
num_rounds: 2
```

Expected outcome:

- threshold aggregation executes correctly
- anomaly metrics are generated

---

### Transformer + FedAvg

```yaml
task: anomaly detection
dataset: metro
model: Transformer
algorithm: fedavg

data_path: ./data/metropt_transformer
win_size: 20
input_c: 14
output_c: 14
```

Expected outcome:

- training completes successfully
- threshold aggregation executes correctly
- anomaly metrics are generated

Important:

The configured win_size must match the window size used when generating the Transformer dataset.

For example, datasets created using:

```bash
python3 data_preprocessing/scripts/prepare_windowed_dataset.py --win_size 20
```

must be trained using:

```yaml
win_size: 20
```

---

### Verification Checklist

After a successful test run:

- Monitoring logs exist under outputs/monitoring/
- Metrics files are generated
- Global model checkpoints are saved (if enabled)
- Inference executes successfully using the saved model
- No client or server exceptions occur during training

---

### Recommended Release Validation Matrix

| Model | Algorithm | Status |
|---------|---------|---------|
| CNN_MNIST | FedAvg | Verified |
| CNN_MNIST | FedAvg+KD | Verified |
| CNN_MNIST | DRFL | Verified |
| Autoencoder | FedAvg | Verified |
| Transformer | FedAvg | Verified |

---

## 🏗️ Project Structure

```
.
├── algorithms/
├── clients/
├── servers/
│   └── strategies/
├── models/
├── data_loading/
├── data_preprocessing/
├── evaluation/
├── monitoring/
├── tests/
├── conf/
├── data/
├── utils/
├── outputs/
│
├── configure.py
├── start_server.py
├── run_inference.py
```

---

## 🧩 Extending the Framework

### Add a new model
- Add it in `models/`
- Register it in GUI

### Add a new dataset
- Add preprocessing/export logic in `data_preprocessing/`
- Define structure in GUI validation

### Add a new algorithm
- Implement client-side logic in:
  - `algorithms/`
- If needed, implement custom server aggregation in:
  - `servers/strategies/`
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

---

## 🖥️ Running in Distributed Environments

The framework supports deployment using Flower's distributed execution model (e.g. SuperLink/SuperNode deployments).

Cluster-specific deployment procedures depend on the target infrastructure and are therefore not included in this repository.

The federated learning workflows described in this README can be executed both locally and in distributed environments supported by Flower.

---

## ⚙️ Deployment-oriented design

The framework is designed as a deployment-oriented federated learning platform for research, experimentation, and pilot deployments. It provides:

  - simulation and real-client execution modes
  - configurable client and server workflows
  - monitoring and KPI collection
  - communication compression
  - model persistence and warm-start training
  - inference and evaluation workflows
  - support for multiple federated learning algorithms

The framework can serve as a foundation for practical federated learning deployments and can be extended with organization-specific security, orchestration, and operational requirements when needed.

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

- lidija.fodor@dmi.uns.ac.rs
- nemanjab4h@gmail.com

---

## ⭐ Final Note

This tool is designed for:
- research reproducibility
- rapid FL experimentation
- bridging simulation and real deployment

