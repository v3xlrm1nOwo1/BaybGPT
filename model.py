import math
import torch
import inspect
import configparser
import torch.nn as nn 
import torch.nn.functional as F



class Config:
    def __init__(self, config_file_path: str, section: str) -> None:
        # Load configuration from config.ini file
        config_parser = configparser.ConfigParser()
        config_parser.read(config_file_path)

        # Model hyperparameters
        self.embed_size = config_parser.getint(section=section, option="embed_size")
        self.heads = config_parser.getint(section=section, option="heads")
        self.feed_forward_size = config_parser.getint(section=section, option="feed_forward_size")
        self.block_size = config_parser.getint(section=section, option="block_size")
        self.padded_vocab_size = config_parser.getint(section=section, option="padded_vocab_size")
        self.num_layer = config_parser.getint(section=section, option="num_layer")
        self.beta_1 = config_parser.getfloat(section=section, option="beta_1")
        self.beta_2 = config_parser.getfloat(section=section, option="beta_2")
        self.eps = config_parser.getfloat(section=section, option="eps")
        self.weight_decay = config_parser.getfloat(section=section, option="weight_decay")
        self.dropout = config_parser.getfloat(section=section, option="dropout")
        self.bias = config_parser.getboolean(section=section, option="bias") 

class MultiHeadCausalAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super(MultiHeadCausalAttention, self).__init__()
        self.config = config
        
        # Ensure the embedding size is divisible by the number of heads for equal division
        assert self.config.embed_size % self.config.heads == 0, "Embedding size needs to be divisible by heads"

        # Define linear transformations for keys, queries, and values
        self.keys = nn.Linear(self.config.embed_size, self.config.embed_size, bias=self.config.bias)
        self.queries = nn.Linear(self.config.embed_size, self.config.embed_size, bias=self.config.bias)
        self.values = nn.Linear(self.config.embed_size, self.config.embed_size, bias=self.config.bias)
        
        # Output projection layer
        self.c_proj_out = nn.Linear(self.config.embed_size, self.config.embed_size)

        # Not really a `Bias`, nore of mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(self.config.block_size, self.config.block_size))
                             .view(1, 1, self.config.block_size, self.config.block_size))
        
        # regularization
        self.attn_dropout = nn.Dropout(self.config.dropout)
        self.resid_dropout = nn.Dropout(self.config.dropout)

        self.c_proj_out.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract dimension: Batch size, Sequence length and Embedding size
        batch_size, seq_len, embed_size = x.size()
        
        # Process queries, keys, values: split into `heads` number of heads, each with head dimension (embed_size/heads)
        # Using .view we divide queries, keys and values for each head
        # Before transpose: [batch_size, seq_len, heads, embed_size/heads]
        # After transpose: [batch_size, heads, seq_len, embed_size/heads]
        all_keys = self.keys(x).view(batch_size, seq_len, self.config.heads, embed_size // self.config.heads).transpose(1, 2)
        all_queries = self.queries(x).view(batch_size, seq_len, self.config.heads, embed_size // self.config.heads).transpose(1, 2)
        all_values = self.values(x).view(batch_size, seq_len, self.config.heads, embed_size // self.config.heads).transpose(1, 2)

        queries_keys = (all_queries @ all_keys.transpose(-1, -2)) * (1.0 / math.sqrt(all_keys.size(-1)))

        # Attention (materializes the larg (seq_len, seq_len) matrix for all the queries and keys)
        masked_queries_keys = queries_keys.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        
        attn_score = F.softmax(masked_queries_keys, dim=-1) 
        attn_score = self.attn_dropout(attn_score)

        out = (attn_score @ all_values).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_size)  

        # Apply the final linear projection layer and output projection
        out = self.c_proj_out(out)  # Transform back to original embed size
        out = self.resid_dropout(out)

        return out

class FeedForward(nn.Module):
    def __init__(self, config: Config) -> None:
        super(FeedForward, self).__init__()
        self.c_fc = nn.Linear(config.embed_size, config.feed_forward_size, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.feed_forward_size, config.embed_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super(Block, self).__init__()
        # Pre-attention layer normalisation
        self.norm_attn = nn.LayerNorm(config.embed_size, bias=config.bias)
        # Pre-feed-forward layer normalisation
        self.norm_ffwd = nn.LayerNorm(config.embed_size, bias=config.bias)        
        # Causal attention mechanism
        self.attention = MultiHeadCausalAttention(config)
        #Fees-forward network (MLP)
        self.feed_forward = FeedForward(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = x + self.attention(self.norm_attn(x))
        ffwd = attn + self.feed_forward(self.norm_ffwd(attn))
        return ffwd
    
class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
    
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(self.config.padded_vocab_size, self.config.embed_size),
                wpe = nn.Embedding(self.config.block_size, self.config.embed_size),
                h = nn.ModuleList([Block(self.config) for _ in range(self.config.num_layer)]),
                ln_f = nn.LayerNorm(self.config.embed_size),
            )
        )
        self.lm_head = nn.Linear(config.embed_size, config.padded_vocab_size, bias=self.config.bias)

        # Weight shearing scheme
        self.transformer.wte.weight = self.lm_head.weight 

        # init params
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.num_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def configure_optimizers(self, learning_rate, device):
        # Start with all of the candidate parameters (the require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"

        optimizer = torch.optim.AdamW(params=optim_groups, lr=learning_rate, betas=(self.config.beta_1, self.config.beta_2), eps=self.config.eps, fused=use_fused)
        return optimizer


    def forward(self, idx: torch.Tensor, targets=None) -> torch.Tensor:
        # Extract dimension: Batch size and Sequence length
        batch_size, seq_len = idx.size()
        assert self.config.block_size >= seq_len, f"Cannot forward sequence of length {seq_len}, block size is only {self.config.block_size}"

        # Forward the token and posistion embedings
        pos = torch.arange(start=0, end=seq_len, dtype=torch.long, device=idx.device) # Shape (seq_len)
        pos_emb = self.transformer.wpe(pos) # Position embeddings of shape (batch_size, embed_size)
        tok_emb = self.transformer.wte(idx) # Token embeddings of shape (batch_size, embed_size)
        
        # Token embeddings + Position embeddings
        x = tok_emb + pos_emb

        # Forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)

        # Forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x) # Shape (batch_size, seq_len, padded_vocab_size)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            loss = None

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, device, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
