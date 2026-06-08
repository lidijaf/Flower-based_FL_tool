from typing import Dict, Any, List

import torch
import torch.nn.functional as F

from algorithms.base import BaseAlgorithm


class FedAvgKDAlgorithm(BaseAlgorithm):
    @property
    def name(self) -> str:
        return "fedavg+KD"

    def fit(self, client, parameters: List, config: Dict[str, Any]):
        client.set_parameters(parameters)

        model_name = client.cfg.get("model", "").lower()
        if "cnn" not in model_name:
            raise ValueError("FedAvg+KD is currently implemented for CNN classification models only.")

        epochs = client.cfg.get("local_epochs", client.cfg.get("epochs", 1))
        lr = client.cfg.get("learning_rate", client.cfg.get("lr", 0.001))
        temperature = client.cfg.get("kd_temperature", 2.0)
        alpha = client.cfg.get("kd_alpha", 0.5)

        student = client.model
        teacher = type(client.model)()
        teacher.load_state_dict(client.model.state_dict())
        teacher.to(client.device)
        teacher.eval()

        optimizer = torch.optim.Adam(student.parameters(), lr=lr)

        student.train()
        total_loss = 0.0
        num_batches = 0

        for _ in range(epochs):
            for x, y in client.trainloader:
                x = x.to(client.device)
                y = y.to(client.device)

                optimizer.zero_grad()

                student_logits = student(x)

                with torch.no_grad():
                    teacher_logits = teacher(x)

                ce_loss = F.cross_entropy(student_logits, y)

                kd_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1),
                    reduction="batchmean",
                ) * (temperature ** 2)

                loss = alpha * ce_loss + (1.0 - alpha) * kd_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        return (
            client.get_parameters(None),
            len(client.trainloader.dataset),
            {
                "train_loss": float(avg_loss),
                "kd_temperature": float(temperature),
                "kd_alpha": float(alpha),
            },
        )

    def evaluate(self, client, parameters: List, config: Dict[str, Any]):
        from models.modelCNN import test_CNN

        client.set_parameters(parameters)

        test_loss, accuracy, precision, recall, f1_score = test_CNN(
            client.model,
            client.testloader,
        )

        return (
            float(test_loss),
            len(client.testloader.dataset),
            {
                "test_loss": float(test_loss),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score),
            },
        )
