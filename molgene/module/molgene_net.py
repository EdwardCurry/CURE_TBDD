
import os
import yaml
import torch
import torch.nn as nn
from module.molgene_encoder import molgene_encoder
from module.hgraph2graph import molgene_decoder
from module.molgene_feat import molgene_feat
from module.molgene_feat_2d import molgene_feat_2d
from module.hgraph2graph.drug_encoder import drug_encoder

class molgene_net(nn.Module):
    def __init__(self, config):
        super(molgene_net, self).__init__()

        # molgene_en_feat
        self.encoder = molgene_encoder(config)
        self.feat = molgene_feat(config=config)

        # smile_encode
        self.drug_encoder = drug_encoder(config)

        self.loss_fn = nn.MSELoss(reduction='mean')


    def drug_embed(self, smiles):
        drug_embed = self.drug_encoder.drug_encode_run(smiles)
        return drug_embed # (1, 250)


    def cell_embed(self, x_org, x_drug):
        x_org_encode = self.encoder(x_org) # (c, h ,w)
        x_drug_encode = self.encoder(x_drug) # (c, h ,w)
        x_embed = self.feat(x_org_encode, x_drug_encode)
        return x_embed

    
    def forward(self, x_org, x_drug, smiles):
        x_embed = self.cell_embed(x_org, x_drug)
        decoded_smiles = self.drug_encoder.drug_reconstruct(x_embed)
        




class molgene_en_feat(nn.Module):
    def __init__(self, config):
        super(molgene_en_feat, self).__init__()

        # molgene_en_feat
        self.encoder = molgene_encoder(config)
        self.feat = molgene_feat(config=config)

        # smile_encode
        self.drug_encoder = drug_encoder(config)

        self.loss_fn = nn.MSELoss(reduction='mean')


    def drug_embed(self, smiles):
        drug_embed = self.drug_encoder.drug_encode_run(smiles)
        return drug_embed # (1, 250)


    def cell_embed(self, x_org, x_drug):
        x_org_encode = self.encoder(x_org) # (c, h ,w)
        x_drug_encode = self.encoder(x_drug) # (c, h ,w)
        x_embed = self.feat(x_org_encode, x_drug_encode)
        return x_embed


    def forward(self, x_org, x_drug, smiles):
        x_embed_ = self.cell_embed(x_org, x_drug)
        drug_embed_ = self.drug_embed(smiles)

        loss = self.loss_fn(x_embed_, drug_embed_)
        return x_embed_, drug_embed_, loss

    def get_embed(self, x_org, x_drug):
        x_embed_ = self.cell_embed(x_org, x_drug)
        return x_embed_






class molgene_only_feat(nn.Module):
    def __init__(self, config):
        super(molgene_only_feat, self).__init__()

        # molgene_en_feat
        # self.encoder = molgene_encoder(config)
        self.feat = molgene_feat_2d(config=config)

        # smile_encode
        self.drug_encoder = drug_encoder(config)

        self.loss_fn = nn.MSELoss(reduction='mean')


    def drug_embed(self, smiles):
        drug_embed = self.drug_encoder.drug_encode_run(smiles)
        return drug_embed # (1, 250)



    def forward(self, x_org, x_drug, smiles):
        x_embed_ = self.feat(x_org, x_drug)
        drug_embed_ = self.drug_embed(smiles)

        loss = self.loss_fn(x_embed_, drug_embed_)
        return x_embed_, drug_embed_, loss

if __name__ == "__main__":
    with open("./config/config.yaml", "r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file)
    model = drug_encoder(config_dict)


