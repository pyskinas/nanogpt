# Core Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Other useful imports
import math                         # Used like once to get cos()
from dataclasses import dataclass   # Make GPTConfig a dataclass
import tiktoken                     # Get gpt2 encoder
import time                         # Time training
import matplotlib.pyplot as plt     # Plots!
import inspect                      # Check if can fuse AdamW
import numpy as np                  # To unpack .npy binaries created by fineweb.py
from hellaswag import render_example, iterate_examples


from torch.distributed import init_process_group, destroy_process_group # Distributing compute over multiple gpus
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# For kaggle
import os
PATH = '/kaggle/input/shakespeare'

def hello_from_gpt_dot_py():
  print("Hello from gpt.py!")

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.NANOGPT_DAMP_FLAG = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)

        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, nh, T, head_dim)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, nh, T, head_dim)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, nh, T, head_dim)

        # # Manual
        # att = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        # Pytorchy
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)        

        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
        
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.NANOGPT_DAMP_FLAG = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
       super().__init__()
       self.config = config

       self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(config.vocab_size, config.n_embd),
        wpe = nn.Embedding(config.block_size, config.n_embd),
        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ln_f = nn.LayerNorm(config.n_embd),
       ))
       self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

       self.transformer.wte.weight = self.lm_head.weight
 
       self.apply(self._init_weights_)

    def _init_weights_(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, "NANOGPT_DAMP_FLAG"):
                std *= (2 * self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=std) 
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # Get params.
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # Get the params that you're going to weight decay. 
        # Don't decay 1D params ~ bias, scale, shift
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Number of decayed tensors: {len(decay_params)} | Number of decayed params: {num_decay_params:,}")
        print(f"Number of nodecay tensors: {len(nodecay_params)} | Number of nodecay params: {num_nodecay_params:,}")
        
        # Can we fuse? What is fusing? idk lol
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device_type
        print(f"Using fused AdamW: {use_fused}")
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas= (0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    
    def forward(self, idx, targets=None):
        # idx will have shape (B, T)
        B, T = idx.shape
        assert T <= self.config.block_size, f"Block size, {T}, too large, max block size is {self.config.block_size}"
        # Need position and token embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # Shape (B, T, vocab_size)
        loss = None

        if targets is not None:
            targets = targets.to(idx.device)
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))

        return logits, loss
    
    # def generate(self, string, batch_size, max_length):
    #     tokens = tokenizer.encode(string)
    #     tokens = tokens.repeat((batch_size,))

    #     for i in range(max_length):
    #         B, T = tokens.shape
    #         logits = self.forward(tokens) # Shape (B, T, vocab_size)
    #         probs = F.softmax(logits, dim=-1)
    #         new_tokens = torch.multinomial(probs[:,-1], 1) # Shape (B, 1)
    #         tokens = torch.cat((tokens, new_tokens), dim=1)
        
    #     return tokenizer.decode(tokens)
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained parameters from GPT2 model on huggingface"""
        assert model_type in {'gpt2','gpt2-medium','gpt2-large','gpt2-xl',}
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt: {model_type}")

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2'          : dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium'   : dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large'    : dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl'       : dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        # Create a from-scratch GPT2 Model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # init huggingface transformers model of GPT2
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring that all parameteres match in name and shape
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 
                      'attn.c_proj.weight',
                      'mlp.c_fc.weight',
                      'mlp.c_proj.weight' 
                      ]
        assert len(sd_keys_hf) == len(sd_keys), f"Mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Conv1D cases
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # All other cases
                assert sd_hf[k].shape == sd[k].shape, sd_hf[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoader:
    def __init__(self, B, T, process, num_processes, split):
        self.B = B
        self.T = T
        self.process = process
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # Get all the filenames
        root_dir = "edu_fineweb10B"
        shards = os.listdir(root_dir)
        shards = [s for s in shards if split in s] # get only train or val
        shards = sorted(shards)
        shards = [os.path.join(root_dir, s) for s in shards]
        self.shards = shards
        assert len(self.shards) > 0, f"Found no shards in directory {root_dir}"
        if master_process:
            print(f"Found {len(shards)} shards for split {split}")

        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = B * T * process
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = B * T * self.process
    
    def next_batch(self):
        B = self.B 
        T = self.T
        p = self.process
        np = self.num_processes

        # Get a batch of tokens, +1 for the targets
        tokens = self.tokens[self.current_position:self.current_position + B * T + 1]
        # Get inputs and targets
        x = tokens[:-1].view(B, T)
        y = tokens[1:].view(B, T)
        # Incrememnt batch count to slide over data
        self.current_position += B * T * np
        # If our next batch won't fit then start at 0 again
        if self.current_position + (B * T * np + 1) > len(self.tokens   ):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * p
        return x, y

class lr_scheduler:
    def __init__(self, min, max, climb_steps, decay_steps):
        self.min = min
        self.max = max
        self.climb_steps = climb_steps
        self.decay_steps = decay_steps

    def get(self, step):
        if step < self.climb_steps:
            return self.max * (step + 1) / self.climb_steps
        elif step > self.climb_steps + self.decay_steps:
            return self.min

        # cosine decay
        decay_ratio = (step - self.climb_steps)/self.decay_steps
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi*decay_ratio))
        return self.min + (self.max - self.min) * coeff

def get_most_likely_row(tokens, mask, logits):
    # Get's the predicted row for hellaswag.
    shift_tokens = (tokens[..., 1:]).contiguous()
    shift_logits = (logits[..., :-1, :]).contiguous()
    flat_shift_tokens = shift_tokens.view(-1)
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    shift_mask = (mask[..., 1:]).contiguous()
    masked_losses = shift_losses * shift_mask
    sum_loss = masked_losses.sum(dim=-1)
    avg_loss = sum_loss / shift_mask.sum(-1)
    pred_norm = avg_loss.argmin().item()
    return pred_norm 
# -----------------------------------------------------------------------
# torchrun --standalone --nproc_per_node=8 gpt.py

# Multiple GPUs
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    # DDP run
    assert torch.cuda.is_available() # Need cuda
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # This process will do logging, checkpointing 
else:
    # Vanilla run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # Set device
    device = "cpu"
    if torch.cuda.is_available():
        device = 'cuda'
    print("Using device:", device)

# Pytorch can be strict with device vs device type
device_type = "cuda" if device.startswith("cuda") else "cpu"

# Initialize data loader
full_batch_size = 524288 # 2**19 ~ 0.5M
warmup_steps = 715
num_steps = 18882 # diff is bc my calc is (1e8*99)/2**19
B, T = 64, 1024
max_lr = 6e-4
min_lr = max_lr * 0.1
lrs = lr_scheduler(min_lr, max_lr, warmup_steps, num_steps) # I failed here first time round bc i put hard coded values for min and max lr
micro_batch_steps = full_batch_size // (B * T * ddp_world_size)
loader = DataLoader(B=B, T=T, process=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoader(B=B, T=T, process=ddp_rank, num_processes=ddp_world_size, split='val')

torch.set_float32_matmul_precision('high')

# Initialize model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)

# Compile the model (?)
use_compile = False
if use_compile:
    model = torch.compile(model)

# No idea.
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# Monitoring
if master_process:
    print("DDP world size:", ddp_world_size)
    print("total batch size:", full_batch_size)
    print("gradient accumulation steps", micro_batch_steps)
    print("Micro Batch Steps:", micro_batch_steps)
    print("Using compiled model." if use_compile else "Model not compiled.")

# Initialise optimizer
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

# Tracking 
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, "w") as f:
    pass

# Train
for step in range(num_steps):
    last_step = step == num_steps - 1

    # Once in a while sample from the model; except 0
    if ((step%250 == 0 and step > 0) or last_step) and (not use_compile):
        model.eval()
        enc = tiktoken.get_encoding("gpt2")
        max_length = 128
        num_generations = 4
        string = "Hello, I'm a language model,"
        tokens = enc.encode(string)
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0) # (1, T)
        tokens = tokens.repeat(num_generations,1) # (num_gen, T)
        x = tokens.to(device)

        while x.size(1) < max_length:
            with torch.no_grad():
                B, T = x.shape
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x) # Shape (B, T, vocab_size)
                probs = F.softmax(logits, dim=-1)
                probs = probs[:,-1]
                topk_probs, topk_ind = torch.topk(probs, 50, dim=-1) # (5,50)
                ix = torch.multinomial(topk_probs, 1) # (5, 1)
                xcol = torch.gather(topk_ind, -1, ix)
                x = torch.cat((x, xcol), dim=1)

        for i in range(num_generations):
            tokens = x[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"{ddp_rank}> {decoded}")
            
    # Once in a while get validation loss every now and then
    if (step%250 == 0 or last_step) and (not use_compile):
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for val_step in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        # Reduce loss across all processes
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"Validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # Optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model' : raw_model.state_dict(),
                    'config' : raw_model.config,
                    'step' : step,
                    'val_loss' : val_loss_accum.item()
                }
                torch.save(checkpoint, checkpoint_path)

    # Once in a while get Hellaswag loss
    if (step%250 == 0 or last_step) and (not use_compile):
        model.eval()
        num_total = 0
        num_correct_norm = 0
        for i, example in enumerate(iterate_examples("val")):
            # For ddp, only do it on that specific rank's turn
            if i % ddp_world_size != ddp_rank:
                continue
            
            # Get the data
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # Reduce stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm/num_total
        if master_process:
            print(f"Hellaswag eval: {num_correct_norm}/{num_total} = {acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # Training
    t0 = time.time()
    model.train()
    optimizer.zero_grad()

    final_loss = 0.0
    for micro_step in range(micro_batch_steps):
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (micro_step == micro_batch_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss  = loss/micro_batch_steps
        final_loss += loss.detach()
        loss.backward()
    
    # Average the loss over all GPUS
    if ddp:
        dist.all_reduce(final_loss, op=dist.ReduceOp.AVG)
        
    # Optimize
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # snip snip
    lr = lrs.get(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    # Tracking
    if device_type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0)*1000 #  in milliseconds
    tokens_per_sec = (loader.B * loader.T * micro_batch_steps * ddp_world_size)/(t1 - t0)
    if master_process:
        print(f"step {step} | train loss {final_loss.item():.4f} | norm {norm:.4f} | {dt:.2f}ms | tok/sec {tokens_per_sec:.0f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {final_loss.item():.6f}\n")

# Make ddp happy
if ddp:
    destroy_process_group()