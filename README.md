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
