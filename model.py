import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import math
import numpy as np


class GATModel(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.2):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = 0.2

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x


class MLPModel(nn.Module):
    def __init__(self, hidden_channels, out_channels, temperature, dropout=0.2):
        super(MLPModel, self).__init__()
        self.source_probs = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )
        self.temperature = temperature

    def forward(self, h, batch):

        logits = self.source_probs(h).squeeze(-1)
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        graph_probs = []
        graph_samples = []
        graph_logits_split = []

        for batch_idx in torch.unique(batch):
            mask = (batch == batch_idx)
            graph_logits = logits[mask]
            graph_logits_split.append(graph_logits)
            probs = F.softmax(graph_logits, dim=0)
            hard_sample = torch.zeros_like(probs)
            hard_sample[torch.argmax(probs)] = 1.0
            graph_probs.append(probs)
            graph_samples.append(hard_sample)

        return {
            'logits': logits,
            'batch': batch,
            'graph_logits': graph_logits_split,
            'graph_probs': graph_probs,
            'graph_samples': graph_samples,
            'features': h
        }


def reparameterize(mu, logvar):

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class VAE_A_Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.1):
        super(VAE_A_Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.W_q = nn.Linear(latent_dim, latent_dim, bias=False)
        self.W_k = nn.Linear(latent_dim, latent_dim, bias=False)

    def forward(self, h):
        h = self.fc1(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        Q = self.W_q(mu)
        K = self.W_k(mu)
        V = mu
        A = Q.T @ K
        A = A / (mu.size(0) ** 0.5)
        A = F.softmax(A, dim=-1)
        mu_new = (A @ V.T).T
        
        if self.training:
            z = reparameterize(mu_new, logvar)
        else:
            z = mu_new
        return z, mu_new, logvar


class VAEEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.1):
        super(VAEEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, h):
        h = self.fc1(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        if self.training:
            z = reparameterize(mu, logvar)
        else:
            z = mu
        return z, mu, logvar


class VAEDecoder(nn.Module):

    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(VAEDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z0):
        return self.net(z0)


def compute_loss(outputs, y, batch):

    total_loss = 0.0
    num_graphs = torch.unique(batch).size(0)
    for i, batch_idx in enumerate(torch.unique(batch)):
        mask = (batch == batch_idx)
        true_labels = y[mask].float()
        assert true_labels.sum() == 1, "Error multi-sources in true label"
        pred_logits = outputs['graph_logits'][i]

        N = len(true_labels)
        pos_weight = min(math.log(N + 1), 5.0)
        weight = torch.full_like(true_labels, 1.0)
        weight[true_labels == 1] = pos_weight

        loss = F.binary_cross_entropy_with_logits(pred_logits, true_labels, weight=weight)
        total_loss += loss

    return total_loss / num_graphs


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x, time_proj):
        residual = x
        x = self.norm(x + time_proj)
        x = self.mlp(x)
        return x + residual


class Diffusion(nn.Module):


    def __init__(self, max_time_steps, beta_start, beta_end, time_embed_dim, latent_dim, hidden_dim):
        super(Diffusion, self).__init__()
        self.max_time_steps = max_time_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.time_embed_dim = time_embed_dim
        self.register_buffer('betas', self.get_beta_schedule())
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_bar', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_bar_prev', torch.cat([torch.tensor([1.0]), self.alphas_bar[:-1]]))
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(self.alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1.0 - self.alphas_bar))
        time_pos_enc = torch.zeros(max_time_steps, time_embed_dim)
        position = torch.arange(0, max_time_steps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, time_embed_dim, 2).float() * (-math.log(10000.0) / time_embed_dim))
        time_pos_enc[:, 0::2] = torch.sin(position * div_term)
        time_pos_enc[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('time_pos_enc', time_pos_enc)  # [T, D]
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.x_proj = nn.Linear(latent_dim, hidden_dim)
        self.time_proj = nn.Linear(time_embed_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(3)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def q_sample(self, z0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)
        sqrt_alphas_bar_t = self.sqrt_alphas_bar[t].unsqueeze(1)
        sqrt_one_minus_alphas_bar_t = self.sqrt_one_minus_alphas_bar[t].unsqueeze(1)
        zt = sqrt_alphas_bar_t * z0 + sqrt_one_minus_alphas_bar_t * noise
        return zt

    def forward(self, z0, t):
        zt = self.q_sample(z0, t)
        time_pos = self.time_pos_enc[t]
        time_emb = self.time_mlp(time_pos)
        time_proj = self.time_proj(time_emb)
        x = self.x_proj(zt)
        for block in self.blocks:
            x = block(x, time_proj)
        x = self.output_norm(x)
        x = F.silu(x)
        z0_recon = self.output_proj(x)
        return z0_recon

    def get_beta_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.max_time_steps)


class KLAnnealer:
    
    def __init__(self, start_weight=0.001, end_weight=0.1, total_epochs=300):
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.total_epochs = total_epochs
        self.decay_factor = (end_weight / start_weight) ** (1.0 / total_epochs)
    
    def get_weight(self, epoch):
        if epoch >= self.total_epochs:
            return self.end_weight
        
        weight = self.start_weight * (self.decay_factor ** epoch)
        return min(weight, self.end_weight)

class PredAnnealer:
    
    def __init__(self, start_weight=1, end_weight=2, total_epochs=300):

        self.start_weight = start_weight
        self.end_weight = end_weight
        self.total_epochs = total_epochs
        self.decay_factor = (end_weight / start_weight) ** (1.0 / total_epochs)
    
    def get_weight(self, epoch):
        if epoch >= self.total_epochs:
            return self.end_weight
        weight = self.start_weight * (self.decay_factor ** epoch)
        return min(weight, self.end_weight)

class CLDMModel(nn.Module):
    """
    CLDM Model
    """

    def __init__(self, model_args):
        super(CLDMModel, self).__init__()
        self.args = model_args
        self.epochs = self.args.epochs
        self.max_num_node = self.args.max_num_node
        self.in_channels = self.args.in_channels
        self.hidden_channels = self.args.hidden_channels
        self.out_channels = self.args.out_channels
        self.heads = self.args.heads
        self.dropout = self.args.dropout
        self.temperature = self.args.temperature
        self.input_dim = self.args.out_channels
        self.encoder_hidden_dim = self.args.hidden_dim
        self.latent_dim = self.args.hidden_dim
        self.kl_start_weight = self.args.kl_start_weight
        self.kl_end_weight = self.args.kl_end_weight
        self.max_time_steps = self.args.max_time_steps
        self.beta_start = self.args.beta_start
        self.beta_end = self.args.beta_end
        self.pred_start_weight = self.args.pred_start_weight
        self.pred_end_weight = self.args.pred_end_weight
        self.time_embed_dim = self.args.time_embed_dim
        self.predictor_dim = self.args.prime_predictor_hidden_dim
        self.aggregation = GATModel(self.in_channels, self.hidden_channels, self.out_channels, self.heads, self.dropout)
        self.encoder = VAE_A_Encoder(self.input_dim, self.encoder_hidden_dim, self.latent_dim)
        self.diffusion = Diffusion(self.max_time_steps, self.beta_start, self.beta_end,
                                   self.time_embed_dim, self.latent_dim, self.predictor_dim)
        self.decoder = VAEDecoder(self.latent_dim, self.encoder_hidden_dim, self.input_dim)
        self.kl_annealer = KLAnnealer(start_weight=self.kl_start_weight, end_weight=self.kl_end_weight, total_epochs=200)
        self.pred_annealer = PredAnnealer(start_weight=self.pred_start_weight, end_weight=self.pred_end_weight, total_epochs=200)
        self.source_predictor = MLPModel(self.hidden_channels, self.out_channels, self.temperature, self.dropout)
        self.initial_parameters()

    def forward(self, x, edge_index, y, batch, epoch):
        h = self.aggregation(x, edge_index)
        z0, mu, logvar = self.encoder(h)
        latent = z0
        cov = torch.cov(latent.T)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = 0.1 * off_diag.abs().mean()
        t = torch.randint(0, self.max_time_steps, (z0.shape[0],), device=x.device).long()
        z_pred = self.diffusion(z0, t)
        h_recon = self.decoder(z_pred)
        outputs = self.source_predictor(h_recon, batch)
        cross_loss = compute_loss(outputs, y, batch)
        pred_loss = ((z0 - z_pred) ** 2).mean(dim=0).sum()
        recon_loss = ((h - h_recon) ** 2).mean(dim=0).sum()
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
        total_loss = cross_loss + recon_loss + cov_loss +  pred_loss + self.kl_annealer.get_weight(epoch) * kl_loss
        return {
            "total_loss": total_loss,
            "cross_loss": cross_loss,
            "pred_loss": pred_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss
        }

    def inference(self, x, edge_index, batch):
        self.eval()
        with torch.no_grad():
            h = self.aggregation(x, edge_index)
            z0, mu, logvar = self.encoder(h)
            h_recon = self.decoder(z0)
            outputs = self.source_predictor(h_recon, batch)
            pred = []
            for graph_sample in outputs['graph_samples']:
                pred.append(graph_sample)
            return pred

    def initial_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
