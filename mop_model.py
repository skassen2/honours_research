# mop_model.py
# Contains all class definitions for the Mixture of Pathways (MoP) model.
# This file can be imported into other scripts to use the MoPModel.

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ---------------------------------------------------
# 1. Model Configuration Class (MoPConfig)
# ---------------------------------------------------
@dataclass
class MoPConfig:
    """Configuration class for the Mixture of Pathways model."""
    input_dim: int = 4
    output_dim: int = 3
    intermediate_dim: int = 64
    layers: List[str] = ("0,16,32", "0,16,32", "0,16,32")
    task_id: str = "iris"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    task_dim: int = 10
    expert_cost_exponent: float = 2.0
    within_expert_dropout_prob: Optional[float] = None
    routing_weight_noise: Optional[float] = None
    dropout_max_prob: Optional[float] = None
    dropout_router_weight_threshold: Optional[float] = None
    flat_expert_knockout_prob: Optional[float] = None

# ---------------------------------------------------
# 2. Model Component Classes
# ---------------------------------------------------

class Expert(nn.Module):
    """An expert in a MoE layer, which can be an identity or a GRU-based network."""
    def __init__(
        self,
        config: MoPConfig,
        hidden_dim: int,
    ):
        super().__init__()
        self.identity = hidden_dim == 0
        if not self.identity:
            self.rnn = nn.GRU(config.intermediate_dim, hidden_dim)
            self.batchnorm = nn.BatchNorm1d(hidden_dim)
            self.relu = nn.ReLU()
            self.dropout = (
                nn.Dropout(config.within_expert_dropout_prob)
                if config.within_expert_dropout_prob is not None
                else nn.Identity()
            )
            self.output_layer = nn.Linear(hidden_dim, config.intermediate_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.identity:
            return x
        else:
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x, _ = self.rnn(x)
            if x.dim() == 3:
                 x = x.squeeze(0)
            if x.shape[0] > 1 and x.var() != 0:
                 x = self.batchnorm(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.output_layer(x)
            return x

class CostBasedRouter(nn.Module):
    """A router that learns to assign weights to experts based on cost."""
    def __init__(
        self,
        config: MoPConfig,
        expert_dims: List[int],
        num_tasks: int,
    ):
        super(CostBasedRouter, self).__init__()
        self.config = config
        self.expert_dims = expert_dims
        self.num_tasks = num_tasks
        self.rnn = nn.GRU(config.intermediate_dim, config.intermediate_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(config.intermediate_dim, len(expert_dims))

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, prev_layer_output: torch.Tensor, task_ids: torch.Tensor):
        logits, _ = self.rnn(prev_layer_output)
        logits = self.relu(logits)
        logits = self.output_layer(logits)
        indices = torch.zeros(
            (logits.shape[0], logits.shape[1], len(self.expert_dims)),
            dtype=torch.long,
            device=logits.device,
        )
        for i in range(len(self.expert_dims)):
            indices[:, :, i] = i
        raw_router_output = F.softmax(logits, dim=-1)
        router_output = raw_router_output.clone()
        
        if len(self.expert_dims) > 1:
            routing_costs = torch.tensor(
                [expert_dim**self.config.expert_cost_exponent for expert_dim in self.expert_dims],
                dtype=logits.dtype,
                device=logits.device,
            )
            task_expert_usage_losses = {}
            expert_usage_loss = torch.einsum("ijk,k->ij", raw_router_output, routing_costs)
            for i in range(self.num_tasks):
                task_mask = task_ids == i
                if task_mask.sum() > 0:
                    task_expert_usage_losses[i] = (expert_usage_loss[task_mask].sum() / task_mask.sum())
                else:
                    task_expert_usage_losses[i] = torch.tensor(0.0, device=logits.device)
            expert_entropy_loss = (-torch.sum(raw_router_output * torch.log(raw_router_output + 1e-10)) / raw_router_output.nelement())
        else:
            task_expert_usage_losses = None
            expert_entropy_loss = None
            
        return (raw_router_output, router_output, indices, task_expert_usage_losses, expert_entropy_loss)

class SparseMoE(nn.Module):
    """A sparse Mixture of Experts layer that combines expert outputs."""
    def __init__(
        self,
        config: MoPConfig,
        expert_dims: List[int],
        num_tasks: int,
    ):
        super(SparseMoE, self).__init__()
        self.router = CostBasedRouter(config, expert_dims, num_tasks)
        self.experts = nn.ModuleList([Expert(config, expert_dim) for expert_dim in expert_dims])

    def forward(self, x: torch.Tensor, task_ids: torch.Tensor):
        (raw_router_output, router_output, indices, task_expert_usage_losses, expert_entropy_loss) = self.router(x, task_ids)
        final_output = torch.zeros_like(x)
        flat_x = x.view(-1, x.size(-1))
        flat_router_output = router_output.view(-1, router_output.size(-1))

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            if not flat_mask.any():
                continue
            expert_input = flat_x[flat_mask]
            expert_output = expert(expert_input)
            gating_scores = flat_router_output[flat_mask, i]
            weighting_output = torch.einsum("i,ij->ij", gating_scores, expert_output)
            final_output.view(-1, final_output.size(-1)).index_add_(0, torch.where(flat_mask)[0], weighting_output.to(final_output.dtype))

        return (final_output, raw_router_output, task_expert_usage_losses, expert_entropy_loss)

class Block(nn.Module):
    """A single block in the MoP model, containing one SparseMoE layer and a LayerNorm."""
    def __init__(
        self,
        config: MoPConfig,
        expert_dims: List[int],
        num_tasks: int,
    ):
        super().__init__()
        self.sparse_moe = SparseMoE(config, expert_dims, num_tasks)
        self.ln = nn.LayerNorm(config.intermediate_dim)

    def forward(self, x: torch.Tensor, task_ids: torch.Tensor):
        moe_outputs = self.sparse_moe(x, task_ids)
        moe_outputs = (self.ln(x + moe_outputs[0]), *moe_outputs[1:])
        return moe_outputs

class MoPModel(nn.Module):
    """The main Mixture of Pathways (MoP) model."""
    def __init__(self, config: MoPConfig):
        super().__init__()
        self.config = config
        self.num_tasks = 1 if config.task_id == "iris" else 82 # Simplified for single-task case
        self.device = torch.device(config.device)
        self.input_layer = nn.Linear(self.config.input_dim, self.config.intermediate_dim)
        self.blocks = nn.ModuleList(
            [Block(self.config, [int(size) for size in layer.split(",")], self.num_tasks) for layer in self.config.layers]
        )
        self.output_layer = nn.Linear(self.config.intermediate_dim, self.config.output_dim)
        self.to(self.config.device)

    # --- THIS IS THE FIX ---
    # The 'enabled' flag is now dynamic. It will only be True if CUDA is available,
    # preventing the warning when running on a CPU.
    @torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available())
    def forward(self, x: torch.Tensor):
        task_ids = torch.zeros(x.shape[0], x.shape[1], dtype=torch.long, device=self.device)
        x = self.input_layer(x)
        total_task_expert_usage_losses = {i: torch.tensor(0.0, device=self.device) for i in range(self.num_tasks)}
        total_expert_entropy_loss = torch.tensor(0.0, device=self.device)

        for block in self.blocks:
            x, _, task_expert_usage_losses, expert_entropy_loss = block(x, task_ids)
            if task_expert_usage_losses is not None:
                for k in task_expert_usage_losses:
                    total_task_expert_usage_losses[k] += task_expert_usage_losses[k]
            if expert_entropy_loss is not None:
                total_expert_entropy_loss += expert_entropy_loss

        x = self.output_layer(x)
        return x, total_task_expert_usage_losses, total_expert_entropy_loss
