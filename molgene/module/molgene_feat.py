

import os
import logging
logging.basicConfig(level=logging.DEBUG)
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Learnable2DPositionEncoding(nn.Module):
    def __init__(self, embed_dim, h, w):
        super().__init__()
        self.h = h
        self.w = w
        self.row_embed = nn.Parameter(torch.randn(h, embed_dim) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(w, embed_dim) * 0.02)
        
    def forward(self, x):

        B, L, C = x.shape
        assert L == self.h * self.w, "not match"
        
        x_2d = x.view(B, self.h, self.w, C)
        
        x_2d = x_2d + self.row_embed.unsqueeze(1)  
        x_2d = x_2d + self.col_embed.unsqueeze(0) 
        
        return x_2d.view(B, L, C)


class molgene_feat_org_hgraph(nn.Module):
    def __init__(self, config=None, input_channels=3, h=256, w=256, embed_dim=128, num_heads=4, decode_embed_dim=2048):
        super().__init__()
        if config:
            print("config")
            input_channels = config["molgene_feat_info"]["input_channels"]
            h = config["molgene_feat_info"]["h"]
            w = config["molgene_feat_info"]["w"]
            embed_dim = config["molgene_feat_info"]["embed_dim"]
            num_heads = config["molgene_feat_info"]["num_heads"]
            decode_embed_dim = config["molgene_feat_info"]["decode_embed_dim"]
        else:
            logging.info("None config found! Please check your config!")
        # Initial projection to embedding dimension
        self.proj = nn.Conv2d(input_channels, embed_dim, kernel_size=1)
        
        # Stacked attention blocks
        self.blocks = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads)
            for _ in range(3)
        ])
        # breakpoint()
        self.pos_encoder = Learnable2DPositionEncoding(embed_dim, h, w)
        
        # Final layers
        self.final_attn = MultiHeadCrossAttention(embed_dim*2, num_heads)
        self.fc = nn.Linear(embed_dim*2 * h * w, decode_embed_dim)
        self.Tanh = nn.Tanh()
        
    def forward(self, a_, b_):
        # Input shape: (1, 3, h, w)
        # Convert to embedding space
        a_emb = self.proj(a_).flatten(2).permute(0, 2, 1)  # (1, h*w, C)
        b_emb = self.proj(b_).flatten(2).permute(0, 2, 1)

        a_emb = self.pos_encoder(a_emb)
        b_emb = self.pos_encoder(b_emb)
        
        # Process through attention blocks
        for block in self.blocks:
            a_emb, b_emb = block(a_emb, b_emb)
        
        # Concatenate and final attention
        combined = torch.cat([a_emb, b_emb], dim=2)  # (1, h*w, 2C)
        out = self.final_attn(combined, combined, combined)
        
        # Flatten and map to final dimension
        out = out.permute(0, 2, 1).flatten(1)  # (1, 2C*h*w)
        out = self.fc(out)
        out = self.Tanh(out)

        return out  # (1, 2048)

class molgene_feat(nn.Module):
    def __init__(self, config=None, input_channels=3, h=256, w=256, embed_dim=128, num_heads=4, decode_embed_dim=2048):
        super().__init__()
        if config:
            print("config")
            input_channels = config["molgene_feat_info"]["input_channels"]
            h = config["molgene_feat_info"]["h"]
            w = config["molgene_feat_info"]["w"]
            embed_dim = config["molgene_feat_info"]["embed_dim"]
            num_heads = config["molgene_feat_info"]["num_heads"]
            # decode_embed_dim = config["molgene_feat_info"]["decode_embed_dim"]
        else:
            logging.info("None config found! Please check your config!")
        # Initial projection to embedding dimension
        self.proj = nn.Conv2d(input_channels, embed_dim, kernel_size=1)
        
        # Stacked attention blocks
        self.blocks = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads)
            for _ in range(3)
        ])
        # breakpoint()
        self.pos_encoder = Learnable2DPositionEncoding(embed_dim, h, w)
        
        # Final layers
        self.final_attn = MultiHeadCrossAttention(embed_dim*2, num_heads)
        self.fc = nn.Linear(embed_dim*2 * h * w, decode_embed_dim)
        self.Tanh = nn.Tanh()
        
    def forward(self, a_, b_):
        # Input shape: (1, 3, h, w)
        # Convert to embedding space
        a_emb = self.proj(a_).flatten(2).permute(0, 2, 1)  # (1, h*w, C)
        b_emb = self.proj(b_).flatten(2).permute(0, 2, 1)

        a_emb = self.pos_encoder(a_emb)
        b_emb = self.pos_encoder(b_emb)
        
        # Process through attention blocks
        for block in self.blocks:
            a_emb, b_emb = block(a_emb, b_emb)
        
        # Concatenate and final attention
        combined = torch.cat([a_emb, b_emb], dim=2)  # (1, h*w, 2C)
        out = self.final_attn(combined, combined, combined)
        
        # Flatten and map to final dimension
        out = out.permute(0, 2, 1).flatten(1)  # (1, 2C*h*w)
        out = self.fc(out)
        # out = self.Tanh(out)
        out = F.relu(out)

        return out  # (1, 2048)

class molgene_feat_simple(nn.Module):
    def __init__(self, b, c=4, h=16, w=16, dropout_prob=0.5): 
        super().__init__()
        fc_input_dim = 2 * c * h * w
        self.fc = nn.Linear(fc_input_dim, 250)
        self.Tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_prob)  
    
    def forward(self, x1, x2):
        # breakpoint()
        x = torch.cat((x1, x2), dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.Tanh(x)
        x = self.dropout(x)  
        return x



# 使用示例
if __name__ == "__main__":
    pass

