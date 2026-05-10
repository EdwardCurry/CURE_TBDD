import os
import logging
logging.basicConfig(level=logging.DEBUG)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCrossAttention_2d(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # QKV projections
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value):
        """
        Args:
            query: (B, N, C)
            key: (B, M, C)
            value: (B, M, C)
        Returns:
            (B, N, C)
        """
        B, N, C = query.shape
        M = key.shape[1]
        
        # Project to Q/K/V
        q = self.Wq(query).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, N, D)
        k = self.Wk(key).view(B, M, self.num_heads, self.head_dim).permute(0, 2, 3, 1)   # (B, H, D, M)
        v = self.Wv(value).view(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, M, D)
        
        # Compute attention scores
        attn = torch.matmul(q, k) / (self.head_dim ** 0.5)  # (B, H, N, M)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, H, N, D)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)  # (B, N, C)
        
        return self.out_proj(out)

class AttentionBlock_2d(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.cross_attn_a = MultiHeadCrossAttention_2d(embed_dim, num_heads)
        self.cross_attn_b = MultiHeadCrossAttention_2d(embed_dim, num_heads)
        self.self_attn = MultiHeadCrossAttention_2d(embed_dim, num_heads)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)
        
    def forward(self, a, b):
        # Cross attention phase
        a_prime = self.cross_attn_a(a, b, b) + a
        a_prime = self.norm1(a_prime)
        
        b_prime = self.cross_attn_b(b, a, a) + b
        b_prime = self.norm2(b_prime)
        
        # Self attention phase
        a_out = self.self_attn(a_prime, a_prime, a_prime) + a_prime
        a_out = self.norm3(a_out)
        
        b_out = self.self_attn(b_prime, b_prime, b_prime) + b_prime
        b_out = self.norm4(b_out)
        
        return a_out, b_out

class Learnable1DPositionEncoding(nn.Module):
    def __init__(self, embed_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.pos_embed = nn.Parameter(torch.randn(seq_len, embed_dim) * 0.02)
        
    def forward(self, x):
        # x: (B, seq_len, C)
        return x + self.pos_embed.unsqueeze(0)

class molgene_feat_2d(nn.Module):
    def __init__(self, config=None, input_channels=1, h=128, w=128, embed_dim=128, num_heads=4, decode_embed_dim=256):
        super().__init__()
        if config:
            input_channels = config["molgene_feat_info"]["input_channels"]
            h = config["molgene_feat_info"]["h"]
            w = config["molgene_feat_info"]["w"]
            embed_dim = config["molgene_feat_info"]["embed_dim"]
            num_heads = config["molgene_feat_info"]["num_heads"]
            decode_embed_dim = config["molgene_feat_info"]["decode_embed_dim"]
        else:
            logging.info("None config found! Using default parameters.")
        
        self.proj = nn.Linear(128, embed_dim)
        
        self.blocks = nn.ModuleList([
            AttentionBlock_2d(embed_dim, num_heads)
            for _ in range(3)
        ])
        
        self.pos_encoder = Learnable1DPositionEncoding(embed_dim, seq_len=h)
        
        self.final_attn = MultiHeadCrossAttention_2d(embed_dim*2, num_heads)
        self.fc = nn.Linear(2 * embed_dim * h, decode_embed_dim)
        
    def forward(self, a_, b_):
        a_ = a_.squeeze(1)  # (B, 128, 128)
        b_ = b_.squeeze(1)
        
        a_emb = self.proj(a_)  # (B, 128, embed_dim)
        b_emb = self.proj(b_)
        
        a_emb = self.pos_encoder(a_emb)
        b_emb = self.pos_encoder(b_emb)
        
        for block in self.blocks:
            a_emb, b_emb = block(a_emb, b_emb)
        
        combined = torch.cat([a_emb, b_emb], dim=2)  # (B, 128, 2*embed_dim)
        out = self.final_attn(combined, combined, combined)
        
        out = out.permute(0, 2, 1).flatten(1)  # (B, 2*embed_dim*128)
        return self.fc(out)

if __name__ == "__main__":
    pass