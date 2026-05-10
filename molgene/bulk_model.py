
import os
import logging
logging.basicConfig(level=logging.DEBUG)
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAlignment(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.align_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.activation = nn.ReLU()
    
    def forward(self, molgene_out, getvec_output):
        mask = (getvec_output > 0).float()
        
        aligned = self.align_conv(molgene_out)
        aligned = self.activation(aligned)
        
        aligned = aligned * mask.unsqueeze(-1)
        
        return aligned

class MultiHeadCrossAttention(nn.Module):
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

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.cross_attn_a = MultiHeadCrossAttention(embed_dim, num_heads)
        self.cross_attn_b = MultiHeadCrossAttention(embed_dim, num_heads)
        self.self_attn = MultiHeadCrossAttention(embed_dim, num_heads)
        
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


class molgene_feat_l1000(nn.Module):
    def __init__(self, config=None, input_len=978, embed_dim=16, num_heads=4, decode_embed_dim=2048, activate='relu'):
        super().__init__()
        if config:
            print("Using config")
            input_len = config["molgene_feat_info"]["input_len"]
            embed_dim = config["molgene_feat_info"]["embed_dim"]
            num_heads = config["molgene_feat_info"]["num_heads"]
            decode_embed_dim = config["molgene_feat_info"]["decode_embed_dim"]
        else:
            logging.info("None config found! Using default parameters")
            
        self.input_len = input_len

        self.proj = nn.Sequential(
            nn.Linear(1, embed_dim), 
            nn.LayerNorm(embed_dim)
        )
        
        self.blocks = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads)
            for _ in range(3)
        ])

        
        self.final_attn = MultiHeadCrossAttention(embed_dim * 2, num_heads)
        self.fc = nn.Linear(2 * embed_dim * input_len, decode_embed_dim)
        self.Tanh = nn.Tanh()
        self.activate = activate
        # self.relu = F.relu()
        
    def forward(self, a_, b_):
        a_ = a_.view(-1, self.input_len, 1)
        b_ = b_.view(-1, self.input_len, 1)
        
        a_emb = self.proj(a_)  # (B, input_len, embed_dim)
        b_emb = self.proj(b_)  # (B, input_len, embed_dim)
        
        # a_emb = self.pos_encoder(a_emb)
        # b_emb = self.pos_encoder(b_emb)
        
        for block in self.blocks:
            a_emb, b_emb = block(a_emb, b_emb)
        
        combined = torch.cat([a_emb, b_emb], dim=2)  # (B, input_len, 2 * embed_dim)
        
        out = self.final_attn(combined, combined, combined)  # (B, input_len, 2 * embed_dim)
        
        out = out.flatten(1)  # (B, 2 * embed_dim * input_len)
        out = self.fc(out)
        if self.activate == "relu":
            out = F.relu(out)
        elif self.activate == "Tanh":
            out = self.Tanh(out)
        return out  # (B, decode_embed_dim)



class TwoHiddenLayerModel(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim1=1024, hidden_dim2=1024, output_dim=2048):
        super(TwoHiddenLayerModel, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.norm1 = nn.BatchNorm1d(hidden_dim1)  
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.norm2 = nn.BatchNorm1d(hidden_dim2) 
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        
        self.output_activation = nn.Tanh()
    

    def forward(self, x):

        x = self.fc1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        x = self.output_activation(x)
        
        return x