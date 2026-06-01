import os
import glob
import time
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from tqdm import tqdm

import numpy as np

import warnings
warnings.filterwarnings("ignore", message="TensorFloat32 tensor cores")

import math

import opt_utils
from optimizers.soap_nanogpt import SOAP
from optimizers.muon_nanogpt import Muon
from optimizers.normuon_nanogpt import NorMuon

from copy import deepcopy

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x



def local_orthogonality_loss(h, window=8, eps=1e-8): # window=4
    B, T, C = h.shape
    h = h / (h.norm(dim=-1, keepdim=True) + eps)  # [B, T, C]
    loss = 0.0
    for k in range(1, window + 1):
        h1 = h[:, :-k, :]  # [B, T-k, C]
        h2 = h[:, k:, :]   # [B, T-k, C]
        loss += (h1 * h2).sum(dim=-1).pow(2).mean()
    return loss / window


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = (1 / (2 * config.n_layer)**0.5)


    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x

    """
    def forward(self, x):
        # Attention
        h_attn = rmsnorm(x)          # [B, T, C]
        ortho_loss_attn = local_orthogonality_loss(h_attn, window=4)
        x = x + self.attn_scale * self.attn(h_attn)

        # MLP
        h_mlp = rmsnorm(x)           # [B, T, C]
        ortho_loss_mlp = local_orthogonality_loss(h_mlp, window=4)
        x = x + self.mlp(h_mlp)

        return x, ortho_loss_attn + ortho_loss_mlp
    """




# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50257
    n_layer : int = 12
    n_head : int = 12
    n_embd : int = 768

class GPT(nn.Module):

    def __init__(self, config, weight_tying=True):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if weight_tying:
            self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None, return_logits=True, eval_mode=False):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device) # shape (t)

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss
 





# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main


import argparse

def parse_args(args):
    parser = argparse.ArgumentParser()


    parser.add_argument("--momentum", type=float, default=0.95) 
     
    parser.add_argument("--adam_lr", type=float, default=None)
    parser.add_argument("--adam_embd_lr", type=float, default=None)
    parser.add_argument("--adam_lm_head_lr", type=float, default=None)
    parser.add_argument("--adam_beta_1", type=float, default=0.9) 
    parser.add_argument("--adam_beta_2", type=float, default=0.95) 
    parser.add_argument("--beta_1", type=float, default=0.9) 
    parser.add_argument("--beta_2", type=float, default=0.95) 

    parser.add_argument("--min_lr_ratio", type=float, default=0.1) 
    
    # Input files
    parser.add_argument('--input_bin', type=str, default='/projects/standard/mhong/shared/glent007/fineweb_data/fineweb10B/fineweb_train_*.bin',
                        help='Input .bin files to train on')
    parser.add_argument('--input_val_bin', type=str, default='/projects/standard/mhong/shared/glent007/fineweb_data/fineweb10B/fineweb_val_*.bin',
                        help='Input .bin files for validation')

    # Optimization hyperparameters
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size in sequences across all devices')
    parser.add_argument('--device_batch_size', type=int, default=64,
                        help='Batch size per device')
    parser.add_argument('--sequence_length', type=int, default=1024,
                        help='Sequence length in tokens')
    parser.add_argument('--num_iterations', type=int, default=6200,
                        help='Number of training iterations')
    parser.add_argument('--warmup_iters', type=int, default=250,
                        help='Number of iterations for linear warmup')
    parser.add_argument('--warmdown_iters', type=int, default=2000,
                        help='Number of iterations for linear warmdown')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay for optimizer')
    parser.add_argument('--adam_weight_decay', type=float, default=0,
                        help='Weight decay for optimizer')
    
    # for other model sizes
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_embd', type=int, default=768)

    # Evaluation and logging
    parser.add_argument('--val_loss_every', type=int, default=125,
                        help='Evaluate validation loss every N steps (0 for only at the end)')
    parser.add_argument('--val_tokens', type=int, default=10485760,
                        help='Number of validation tokens to evaluate')
    parser.add_argument('--save_every', type=int, default=0,
                        help='Save checkpoint every N steps (0 for only at the end)')

    parser.add_argument("--wandb_project_name", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--save_dir", type=str, default=None)
    
    
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--use_nanogpt_weight_tying", default=False, action="store_true")
    parser.add_argument("--scheduler", type=str, default="wsd", choices=["cosine", "wsd_linear_decay", "wsd_expo_decay"])
    parser.add_argument("--seed", type=int, default=0)


    # For EMA-Nesterov
    parser.add_argument("--use_ema_nesterov", default=False, action="store_true")
    parser.add_argument("--use_nesterov_step", default=False, action="store_true")
    parser.add_argument("--lookahead_stepsize", type=float, default=0.5)
    parser.add_argument("--lookahead_ema", type=float, default=0.99)
    parser.add_argument("--ema_nesterov_warmup", type=int, default=0)
    parser.add_argument("--ema_nesterov_rest", type=int, default=0)
    parser.add_argument("--eval_lerp_models", default=False, action="store_true")
    parser.add_argument("--eval_lerp_every", type=int, default=0)
    parser.add_argument("--eval_lerp_gap", type=int, default=1)

    # For GPA
    parser.add_argument("--use_gpa", default=False, action="store_true")
    parser.add_argument("--momentum_y", type=float, default=0.99)

    # For SNOO
    parser.add_argument("--use_snoo", default=False, action="store_true")
    parser.add_argument("--snoo_lr", type=float, default=0.5)
    parser.add_argument("--snoo_momentum", type=float, default=0.25)
    parser.add_argument("--local_steps_K", type=int, default=10)

    # For Lookahead
    parser.add_argument("--use_lookahead", default=False, action="store_true")
    parser.add_argument("--lookahead_step_size", type=float, default=0.99)

    args = parser.parse_args(args)
    return args



def main(args):
    opt_utils.set_seed_deterministic(args.seed)

    # set up DDP (distributed data parallel). torchrun sets this env variable
    assert torch.cuda.is_available()
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    print(f"using device: {device}")
    master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

    if master_process:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, name=args.model_name)

    # convenience variables
    B, T = args.device_batch_size, args.sequence_length
    # calculate the number of steps to take in the val loop.
    assert args.val_tokens % (B * T * ddp_world_size) == 0
    val_steps = args.val_tokens // (B * T * ddp_world_size)
    # calculate the steps of gradient accumulation required to attain the desired global batch size.
    assert args.batch_size % (B * ddp_world_size) == 0
    train_accumulation_steps = args.batch_size // (B * ddp_world_size)

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
    if master_process:
        print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
        print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
    x, y = train_loader.next_batch()

    # init the model from scratch
    num_vocab = 50257
    model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd), args.use_nanogpt_weight_tying)
    model = model.cuda()
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True # suggested by @Chillee
    model = torch.compile(model)
    # here we wrap model into DDP container
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module # always contains the "raw" unwrapped model
    if args.eval_lerp_models:
        if master_process:
            wandb.define_metric("lerp/iterations", hidden=True)
            wandb.define_metric("lerp/eval_loss", step_metric='lerp/iterations')
            wandb.define_metric("lerp/eval_perplexity", step_metric='lerp/iterations')
            wandb.define_metric("nes_extp/iterations", hidden=True)
            wandb.define_metric("nes_extp/eval_loss", step_metric='nes_extp/iterations')
            wandb.define_metric("nes_extp/eval_perplexity", step_metric='nes_extp/iterations')
        prev_model_state = deepcopy(model.state_dict())
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    # init the optimizer(s)
    if args.optimizer == "muon" or args.optimizer == "normuon":
        if args.optimizer == "muon":
            Muon_Class = Muon
        else:
            Muon_Class = NorMuon
        if not args.use_nanogpt_weight_tying:
            optimizer1 = torch.optim.AdamW(raw_model.transformer.wte.parameters(), lr=args.adam_embd_lr, betas=(args.adam_beta_1, args.adam_beta_2), weight_decay=args.weight_decay, fused=True)
            optimizer2 = Muon_Class(raw_model.transformer.h.parameters(), lr=args.lr, momentum=args.momentum)
            optimizer3 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.adam_lm_head_lr, betas=(args.adam_beta_1, args.adam_beta_2), weight_decay=args.weight_decay, fused=True)
            optimizers = [optimizer1, optimizer2, optimizer3]
        else:
            optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.adam_lr, betas=(args.adam_beta_1, args.adam_beta_2), weight_decay=args.weight_decay, fused=True)
            optimizer2 = Muon_Class(raw_model.transformer.h.parameters(), lr=args.lr, momentum=args.momentum)
            optimizers = [optimizer1, optimizer2]
    elif args.optimizer == "adam":
        optimizers = [torch.optim.AdamW(raw_model.parameters(), lr=args.lr, betas=(args.adam_beta_1, args.adam_beta_2), weight_decay=args.weight_decay, fused=True)]
    elif args.optimizer == "soap":
        optimizers = [SOAP(raw_model.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2), precondition_frequency=10, weight_decay=args.weight_decay)]

    if args.use_ema_nesterov:
        trainable_params = [dict(params=raw_model.lm_head.parameters()), 
                            dict(params=raw_model.transformer.h.parameters())]
        inner_optimizers = optimizers
        optimizers = [opt_utils.EMA_Nesterov(
            trainable_params,
            inner_optimizers,
            lookahead_stepsize=args.lookahead_stepsize,
            use_scheduled_lookahead_stepsize=True,
            lookahead_ema=args.lookahead_ema,
            warmup_step=args.ema_nesterov_warmup,
            rest_step=args.num_iterations - args.ema_nesterov_rest,
        )]
    elif args.use_gpa:
        trainable_params = [dict(params=raw_model.lm_head.parameters()), 
                            dict(params=raw_model.transformer.h.parameters())]
        inner_optimizers = optimizers
        optimizers = [opt_utils.GPA(
            trainable_params,
            inner_optimizers,
            momentum_x=args.momentum_y**(1/args.local_steps_K),
            momentum_y=args.momentum_y,
        )]
    elif args.use_snoo:
        trainable_params = [dict(params=raw_model.lm_head.parameters()), 
                            dict(params=raw_model.transformer.h.parameters())]
        inner_optimizers = optimizers
        optimizers = [opt_utils.SNOO(
            trainable_params,
            inner_optimizers,
            lr=args.snoo_lr,
            momentum=args.snoo_momentum,
            local_steps_K=args.local_steps_K,
        )]
    elif args.use_lookahead:
        trainable_params = [dict(params=raw_model.lm_head.parameters()), 
                            dict(params=raw_model.transformer.h.parameters())]
        inner_optimizers = optimizers
        optimizers = [opt_utils.Lookahead(
            trainable_params,
            inner_optimizers,
            lookahead_step_size=args.lookahead_step_size,
            local_steps_K=args.local_steps_K,
        )]




    # learning rate decay scheduler (linear warmup and decay)
    def wsd_lr_lambda(it, total_steps, warmup_steps, decay_steps, min_lr_ratio=0.1, decay_type="linear"):
        assert it <= total_steps
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return (it+1) / warmup_steps
        # 2) constant lr for a while
        elif it < total_steps - decay_steps:
            return 1.0
        else:
            if decay_type == "linear":
                # 3) linear decay
                decay_ratio = (total_steps - it) / decay_steps
                return min_lr_ratio + (1.0 - min_lr_ratio) * decay_ratio
            else:
                # 3) exponential decay
                progress = 1 - (total_steps - it) / decay_steps
                k = -math.log(min_lr_ratio)
                return math.exp(-k * progress)
        
    def cosine_lr_lambda(it, total_steps, warmup_steps, min_lr_ratio=0.1):
        """
        Cosine learning rate schedule with linear warmup and a minimum learning rate ratio.
        Returns a scaling factor (multiplier) for LambdaLR.
        """
        if it < warmup_steps:
            # Linear warmup
            return (it + 1) / warmup_steps
        else:
            # Cosine decay
            progress = (it - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay        
  

    schedulers = []
    if args.use_ema_nesterov or args.use_gpa or args.use_snoo or args.use_lookahead:
        sche_opts = inner_optimizers
    else:
        sche_opts = optimizers
    for opt in sche_opts:
        if args.scheduler == "cosine":
            sche = torch.optim.lr_scheduler.LambdaLR(opt, lambda x: cosine_lr_lambda(x, args.num_iterations, args.warmup_iters))
        elif "wsd" in args.scheduler:
            if "linear" in args.scheduler:
                decay_type = "linear"
            else:
                decay_type = "exponential"
            sche = torch.optim.lr_scheduler.LambdaLR(opt, lambda x: wsd_lr_lambda(x, args.num_iterations, args.warmup_iters, args.warmdown_iters, min_lr_ratio=0, decay_type=decay_type))
        schedulers.append(sche)
            
    if master_process:
        # Initialize wandb
        run_config = dict(vars(args))
        '''
        run_config.update(
            {
                "max_lr": run_config.pop(
                    "lr"
                ),  # rename lr to max_lr to avoid conflicts with scheduler
                #"total_params_M": n_total_params / 1_000_000,
                #"dataset": "allenai/c4",
                #"model": model_config.to_dict(),
                "world_size": ddp_world_size,
                "device": str(device),
            }
        )
        '''
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now")  # save current script

        pbar = tqdm(
            total=args.num_iterations, desc="Update steps", ncols=80
        )


    def evaluate(model, val_loader, val_steps):
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with torch.no_grad(): # of course, we'd like to use ctx here too, but that creates a torch.compile error for some reason
                _, loss = model(x_val, y_val, return_logits=False, eval_mode=True ) # , lambda_ortho=0.0
                val_loss += loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        return val_loss
    

    training_time_ms = 0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.time()
    # begin training
    train_loader.reset()
    for step in range(args.num_iterations + 1):
        wandb_logs = {}
        last_step = (step == args.num_iterations)
        # This effectively ignores timing first 10 steps, which are slower for weird reasons.
        # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
        # steps with dummy data first, and then re-initialize the model and reset the loader.
        if step == 10:
            training_time_ms = 0
            t0 = time.time()
        timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

        # once in a while evaluate the validation dataset
        if (last_step or step == 0 or (args.val_loss_every > 0 and step % args.val_loss_every == 0 and step !=0 )   ): # !!!
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)
            # run validation batches
            val_loss = evaluate(model, val_loader, val_steps)

            # log val loss to console and to logfile
            if master_process:
                print(f' step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.time()
            
            if master_process:
                wandb_logs["final_eval_loss"] = val_loss.cpu().item()
                wandb_logs["final_eval_perplexity"] = np.exp(val_loss.cpu().item())
                        #"final_eval_tokens": evaluated_on_tokens,
                wandb_logs["training_time_ms"] = training_time_ms
                wandb_logs["iterations"] = step
                if not args.eval_lerp_models:
                    wandb.log(
                        wandb_logs,
                        step=step,
                    )
                else:
                    wandb.log(
                        wandb_logs,
                        # step=step,
                    )
                wandb_logs = {}
        
        # run evaluation on linear interpolations
        if step > 0 and args.eval_lerp_models:
                
            if step % args.eval_lerp_every == 0:
                sd_1 = prev_model_state
                sd_2 = deepcopy(model.state_dict())
                # torch.cuda.empty_cache()
                lerp_val_losses = []
                lerp_pos = []
                for a in torch.arange(-1, 2.01, 0.25):
                    state_dict = {}
                    for k in sd_1:
                        state_dict[k] = sd_2[k] + a * (sd_2[k] - sd_1[k])
                    model.load_state_dict(state_dict)
                    lerp_val_losses.append( evaluate(model, val_loader, val_steps) )
                    lerp_pos.append( step + a * args.eval_lerp_gap )
                # revert to current model
                model.load_state_dict(sd_2)

                nes_extp_val_losses = []
                extp_pos = []
                if args.use_ema_nesterov:
                    for a in torch.arange(-1, 2.01, 0.25):
                        optimizers[0].lookahead_step(a)
                        nes_extp_val_losses.append( evaluate(model, val_loader, val_steps) )
                        extp_pos.append(step + a)
                        # revert to current model
                        model.load_state_dict(sd_2)

                
                if master_process:
                    for los, pos in zip(lerp_val_losses, lerp_pos):
                        wandb.log(
                            {"lerp/eval_loss": los.cpu().item(),
                            "lerp/eval_perplexity": np.exp(los.cpu().item()),
                            "lerp/iterations": pos.cpu().item()})
                    
                    for los, pos in zip(nes_extp_val_losses, extp_pos):
                        wandb.log(
                            {"nes_extp/eval_loss": los.cpu().item(),
                            "nes_extp/eval_perplexity": np.exp(los.cpu().item()),
                            "nes_extp/iterations": pos.cpu().item()})
            
            if step % args.eval_lerp_gap == 0:
                prev_model_state = deepcopy(model.state_dict())
        
        if master_process and not args.save_dir is None and (step % args.save_every == 0 or last_step):
            if not os.path.exists(args.save_dir):
                os.mkdir(args.save_dir)
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model_{}.pt".format(step)))
        
            
        # bit confusing: we want to make sure to eval on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break
        
        if args.use_ema_nesterov or args.use_gpa or args.use_snoo or args.use_lookahead:
            lrs = [opt.param_groups[0]["lr"] for opt in optimizers[0].inner_optimizer]
        else:
            lrs = [opt.param_groups[0]["lr"] for opt in optimizers]

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        if args.use_nesterov_step:
            optimizers[0].nesterov_step()
        for i in range(1, train_accumulation_steps+1):
            # forward pass

            with ctx:
                _, loss = model(x, y, return_logits=False)
                train_loss = loss.detach()
            # advance the dataset for the next batch
            x, y = train_loader.next_batch()
            # backward pass
            if i < train_accumulation_steps:
                with model.no_sync(): # there's no need to sync gradients every accumulation step
                    loss.backward()
            else:
                loss.backward() # just sync on the last step
        for p in model.parameters():
            p.grad /= train_accumulation_steps

        # step the optimizers and schedulers
        for opt in optimizers:
            opt.step()
        for sched in schedulers:
            sched.step()
            

        # null the gradients
        model.zero_grad(set_to_none=True)
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
        if master_process:
            approx_time = training_time_ms + 1000 * (time.time() - t0)
            wandb_logs["loss"] = train_loss.cpu().item()
            wandb_logs["training_time"] = approx_time
            wandb_logs["iterations"] = step+1
            for i in range(len(lrs)):
                wandb_logs["opt{}_lr".format(i+1)] = lrs[i]
            if not args.eval_lerp_models:
                wandb.log(
                    wandb_logs,
                    step=step+1,
                )
            else:
                wandb.log(
                    wandb_logs,
                    # step=step+1,
                )

            pbar.update(1)
        
        

    if master_process:
        print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
        pbar.close()

    # -------------------------------------------------------------------------
    # clean up nice
    dist.destroy_process_group()


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)