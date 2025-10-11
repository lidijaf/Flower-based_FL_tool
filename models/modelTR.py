import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import get_cfg

import numpy as np
import math
import os

# EMBEDDING COMPONENT (TOKEN & POSITIONAL EMBEDDING: RAW DATA -> X)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


# MULTI-HEAD SELF-ATTENTION BLOCK COMPONENT

class TriangularCausalMask():  # Forces attention to depend only on past data if needed.
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):  # computes ques, keys, values, and sigmas
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False, device="cpu"):
        super(AnomalyAttention, self).__init__()
        self.device = device
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size
        #self.distances = torch.zeros((window_size, window_size)).cuda()
        self.distances = torch.zeros((window_size, window_size), device=self.device)
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)

    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).to(self.device)
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):  # wraps around AnomalyAttention
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma


# MAIN MODEL

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Embedding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # Encoding
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]


# TRAIN AND TEST FUNCTIONS

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def vali(model: nn.Module, valloader: DataLoader, trainloader: DataLoader, k: float, win_size: int, cfg):
    """
    Validate the model on a given validation dataset.

    Args:
        model (nn.Module): The trained model.
        valloader (DataLoader): DataLoader for validation data.
        trainloader (DataLoader): DataLoader for training data.
        k (float): Weighting factor for series and prior losses.
        win_size (int): Window size for Anomaly Transformer.

    Returns:
        tuple: (average validation loss, threshold value)
    """
    print("======================VALIDATION MODE======================")

    #cfg = get_cfg()
    device = torch.device(cfg.get("device"))
    criterion = nn.MSELoss(reduction='none')
    temperature = 100
    anomaly_ratio = 2

    model.eval()
    model.to(device)

    # === (1) Calculate attention energy statistics on train loader ===
    attens_energy = []
    for input_data, labels in trainloader:
        input = input_data.float().to(device)
        output, series, prior, _ = model(input)

        rec_loss = torch.mean(criterion(input, output), dim=-1)  # shape: (batch_size, seq_len) # mean over features??

        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, win_size)).detach()) * temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, win_size)),
                    series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, win_size)).detach()) * temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, win_size)),
                    series[u].detach()) * temperature

        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = metric * rec_loss
        cri = cri.detach().cpu().numpy()
        attens_energy.append(cri)

    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    train_energy = np.array(attens_energy)

    # === (2) Calculate attention energy on test loader ===
    attens_energy = []
    for input_data, labels in valloader:
        input = input_data.float().to(device)
        output, series, prior, _ = model(input)

        loss = torch.mean(criterion(input, output), dim=-1)

        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, win_size)).detach()) * temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, win_size)),
                    series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, win_size)).detach()) * temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, win_size)),
                    series[u].detach()) * temperature

        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = (metric * loss).detach().cpu().numpy()
        attens_energy.append(cri)

    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    vali_energy = np.array(attens_energy)

    # === (3) Calculate threshold ===
    combined_energy = np.concatenate([train_energy, vali_energy], axis=0)
    threshold = np.percentile(combined_energy, 100 - anomaly_ratio)

    # === (4) Return threshold ===
    return float(threshold)


def train(model: nn.Module, trainloader: DataLoader, valloader: DataLoader, k: float, win_size: int, cfg):
    """
    Train a neural network model for anomaly detection.

    Args:
        model (nn.Module): The model to train.
        trainloader: DataLoader for training data.
        k (float): Weighting factor for series and prior losses.
        win_size (int): Window size for the anomaly transformer.
    """
    print("====================== TRAIN MODE ======================")

    # Load configuration
    #cfg = get_cfg()
    algorithm = cfg.get("algorithm")
    device = torch.device(cfg.get("device"))
    epochs = cfg.get("epochs")

    # Optimizer & loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("learning_rate"))
    criterion = nn.MSELoss()

    # Move model to device
    model.to(device)

    if algorithm == "fedavg":
        model.train()
        rec_loss_list = []

        for epoch in range(epochs):
            for input_data, labels in trainloader:

                # Move data to device
                input_data = input_data.float().to(device)

                # Forward pass
                output, series, prior, _ = model(input_data)

                # Compute losses
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = criterion(output, input_data)

                # Log losses
                rec_loss_list.append(rec_loss.item())

                # loss1: Maximize discrepancy - optimize weights to output series farthest from prior
                loss1 = rec_loss - k * series_loss
                # loss2: Minimize discrepancy - optimize weights to output prior closest to series
                loss2 = rec_loss + k * prior_loss

                # Minimax strategy
                optimizer.zero_grad()
                loss1.backward(retain_graph=True)
                loss2.backward()
                optimizer.step()

        train_rec_loss = np.average(rec_loss_list)
        threshold = vali(model, valloader, trainloader, k, win_size, cfg)

        return train_rec_loss, threshold

    elif algorithm == "pfedme":

        # Copy the parameters obtained from the server (global model),
        # this is done because of the penalty term (mozda obrisati)
        global_params = [val.detach().clone() for val in model.parameters()]
        model.train()
        model.to(device)

        # pfedme-specific configurations
        local_rounds = cfg["config_fit_pfedme"].get("local_rounds")
        local_iterations = cfg["config_fit_pfedme"].get("local_iterations")
        lambda_reg = cfg["config_fit_pfedme"].get("lambda_reg")
        mu = cfg["config_fit_pfedme"].get("mu")
        new = cfg["config_fit_pfedme"].get("new")

        total_loss = 0.0
        count = 0

        for _ in range(local_rounds):
            if not new:
                data_iterator = iter(trainloader)
                input_data, _ = next(data_iterator)
                # Move data to device
                input_data = input_data.to(device)

                for _ in range(local_iterations):
                    # Forward pass
                    output, series, prior, _ = model(input_data)

                    # Compute losses
                    series_loss = 0.0
                    prior_loss = 0.0
                    for u in range(len(prior)):
                        normalized_prior = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(
                            1, 1, 1, win_size
                        )
                        series_loss += (torch.mean(my_kl_loss(series[u], normalized_prior.detach())) +
                                        torch.mean(my_kl_loss(normalized_prior.detach(), series[u])))
                        prior_loss += (torch.mean(my_kl_loss(normalized_prior, series[u].detach())) +
                                       torch.mean(my_kl_loss(series[u].detach(), normalized_prior)))

                    # Normalize losses
                    series_loss = series_loss / len(prior)
                    prior_loss = prior_loss / len(prior)

                    rec_loss = criterion(output, input_data)
                    loss1 = rec_loss - k * series_loss
                    loss2 = rec_loss + k * prior_loss

                    penalty_term = sum((w - w0).norm(2) ** 2 for w, w0 in zip(model.parameters(), global_params))
                    penalized_loss1 = loss1 + (lambda_reg / 2) * penalty_term   # Maximize
                    penalized_loss2 = loss2 + (lambda_reg / 2) * penalty_term   # Minimize

                    optimizer.zero_grad()
                    penalized_loss1.backward(retain_graph=True)
                    penalized_loss2.backward()
                    optimizer.step()

                    total_loss += loss1.item()  # Should it follow penalised loss ?
                    count += 1

                else:
                    for _ in range(local_iterations):
                        data_iterator = iter(trainloader)
                        input_data, _ = next(data_iterator)
                        input_data = input_data.to(device)

                        output, series, prior, _ = model(input_data)

                        series_loss = 0.0
                        prior_loss = 0.0
                        for u in range(len(prior)):
                            norm_prior = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(
                                1, 1, 1, win_size
                            ).detach()

                            series_loss += (
                                    torch.mean(my_kl_loss(series[u], norm_prior)) +
                                    torch.mean(my_kl_loss(norm_prior, series[u]))
                            )
                            prior_loss += (
                                    torch.mean(my_kl_loss(norm_prior, series[u].detach())) +
                                    torch.mean(my_kl_loss(series[u].detach(), norm_prior))
                            )

                        series_loss /= len(prior)
                        prior_loss /= len(prior)

                        rec_loss = criterion(output, input_data)
                        loss1 = rec_loss - k * series_loss
                        loss2 = rec_loss + k * prior_loss

                        penalty_term = sum((w - w0).norm(2) ** 2 for w, w0 in zip(model.parameters(), global_params))
                        penalized_loss1 = loss1 + (lambda_reg / 2) * penalty_term
                        penalized_loss2 = loss2 + (lambda_reg / 2) * penalty_term

                        optimizer.zero_grad()
                        penalized_loss1.backward(retain_graph=True)
                        penalized_loss2.backward()
                        optimizer.step()

                        total_loss += loss1.item()
                        count += 1

                # Update global parameters
                with torch.no_grad():
                    for param, global_param in zip(model.parameters(), global_params):
                        global_param -= mu * lambda_reg * (global_param - param)

        avg_loss = total_loss / count if count > 0 else 0.0
        return global_params, avg_loss

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def test(model: nn.Module, testloader: DataLoader, threshold, win_size: int, cfg):
    print("======================TEST MODE======================")
    #cfg = get_cfg()
    device = torch.device(cfg.get("device"))
    model.eval()
    temperature = 100
    criterion = nn.MSELoss(reduction="none")

    rec_loss_list = []
    attens_energy = []
    test_labels = []

    for input_data, labels in testloader:
        input = input_data.float().to(device)
        output, series, prior, _ = model(input)

        loss = torch.mean(criterion(input, output), dim=-1)

        # Compute rec losses
        rec_loss = torch.mean(loss)
        rec_loss_list.append(rec_loss.item())

        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, win_size)).detach()) * temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, win_size)),
                    series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, win_size)).detach()) * temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, win_size)),
                    series[u].detach()) * temperature

        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = metric * loss
        cri = cri.detach().cpu().numpy()
        attens_energy.append(cri)
        test_labels.append(labels)

    # flattened into raw labels 1D list
    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    test_energy = np.array(attens_energy)
    test_labels = np.array(test_labels)

    output_dir = "C:\\Users\\neman\\PycharmProjects\\tool_demo\\outputs"
    os.makedirs(output_dir, exist_ok=True)  # Creates the folder if it doesn't exist

    save_path = os.path.join(output_dir, "anomaly_outputs.npz")
    np.savez_compressed(save_path, energy=test_energy, labels=test_labels)


    pred = (test_energy > threshold).astype(int)

    gt = test_labels.astype(int)  # ground truth

    print("Final shapes - pred:", pred.shape, "gt:", gt.shape)


    # Detection adjustment
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1


    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    accuracy = accuracy_score(gt, pred)  # percentage of matching labels
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')

    print(f"Accuracy : {accuracy:.4f}, Precision : {precision:.4f}, Recall : {recall:.4f}, F-score : {f_score:.4f}")

    return np.average(rec_loss_list), accuracy, precision, recall, f_score
    

