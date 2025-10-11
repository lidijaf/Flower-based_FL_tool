from torchvision import datasets, transforms
import torch
import os

clients = ["client1", "client2"]
base_path = "mnist/"

transform = transforms.Compose([transforms.ToTensor()])

for client in clients:
    client_path = os.path.join(base_path, client)
    os.makedirs(client_path, exist_ok=True)
    
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    torch.save(train_dataset, os.path.join(client_path, "train.pt"))
    torch.save(test_dataset, os.path.join(client_path, "test.pt"))

