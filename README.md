# Flower-based_FL_tool

The Flower-based FL tool provides a convenient way to run a Federated Learnign (FL) task.

## Setting up the training
- Use gui.py to select the training properties or fill manually the files conf/config_common.yaml, conf/config_server.yaml, conf/config_client.yaml
  
## Instalation and run

- The project requires Python 3.10 or Python 3.11.
- Create a virtual environment and activate it (recommended, but optional)
- install the requirements: `pip install -r requirements.txt`
- Start the server as: `python3 run.py`
- Start the clients as e.g. (choose a client type): `python3 -m clients.clientTR --client_id 1 --data_path ./data/mnist/client1`

## Running on a cluster

- use and adjust the provided scripts
- example with SLURM:
  - run the server: `python slurm-launch.py   --exp-name flower_server   --num-nodes 1   --partition all   --load-env "source ~/.bashrc && eval \"\$(conda shell.bash hook)\" && conda activate flower_env"   --command "python run.py"`
  - run the clients as: `bash launch_all_cleints.sh`  
