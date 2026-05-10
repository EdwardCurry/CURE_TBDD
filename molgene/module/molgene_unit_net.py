import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from module.hgraph2graph.hgraph.mol_graph import MolGraph
from module.hgraph2graph.hgraph.encoder import HierMPNEncoder
from module.hgraph2graph.hgraph.decoder import HierMPNDecoder
from module.hgraph2graph.hgraph.nnutils import *
from module.molgene_net import *
from module.hgraph2graph.hgraph import *
from collections import OrderedDict

def make_cuda(tensors):
    tree_tensors, graph_tensors = tensors
    make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x)
    tree_tensors = [make_tensor(x).cuda().long() for x in tree_tensors[:-1]] + [tree_tensors[-1]]
    graph_tensors = [make_tensor(x).cuda().long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    return tree_tensors, graph_tensors


class Hierdecoder_self(nn.Module):

    def __init__(self, config):
        super(Hierdecoder_self, self).__init__()
        vocab = [x.strip("\r\n ").split() for x in open(config["unit_train"]["vocab"])] 
        self.vocab = PairVocab(vocab)
        self.decoder = HierMPNDecoder(self.vocab, common_atom_vocab, config["unit_train"]["rnn_type"], 
                        config["unit_train"]["embed_size"], config["unit_train"]["hidden_size"], config["unit_train"]["latent_size"], 
                        config["unit_train"]["diterT"], config["unit_train"]["diterG"], config["unit_train"]["dropout"])
        self.latent_size = config["unit_train"]["latent_size"]
        self.R_mean = nn.Linear(config["unit_train"]["hidden_size"], self.latent_size)
        self.R_var = nn.Linear(config["unit_train"]["hidden_size"], self.latent_size)

        # molgene_en_feat
        self.encoder = molgene_encoder(config)
        self.feat = molgene_feat(config=config)

        # self.embed_model = molgene_en_feat(config)

    def rsample(self, z_vecs, W_mean, W_var, perturb=True):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs( W_var(z_vecs) )
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean)  # .cuda()
        # breakpoint()
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon if perturb else z_mean
        return z_vecs, kl_loss

    def sample(self, batch_size, greedy):
        root_vecs = torch.randn(batch_size, self.latent_size).cuda()
        return self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=greedy, max_decode_step=150)

       
    def forward(self, org_img, drug_img, graphs, tensors, orders, perturb_z=True):  #batches, tensors, all_orders
        tree_tensors, graph_tensors = tensors = make_cuda(tensors)

        # root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)

        # root_vecs = self.embed_model.get_embed(org_img, drug_img)
        x_org_encode = self.encoder(org_img) # (c, h ,w)
        x_drug_encode = self.encoder(drug_img) # (c, h ,w)
        root_vecs = self.feat(x_org_encode, x_drug_encode)

        root_vecs, root_kl = self.rsample(root_vecs, self.R_mean, self.R_var, perturb_z)
        kl_div = root_kl

        loss, wacc, iacc, tacc, sacc = self.decoder((root_vecs, root_vecs, root_vecs), graphs, tensors, orders)
        return loss, kl_div.item(), wacc, iacc, tacc, sacc

    def get_embed(self, org_img, drug_img):
        x_org_encode = self.encoder(org_img) # (c, h ,w)
        x_drug_encode = self.encoder(drug_img) # (c, h ,w)
        org_root_vecs = self.feat(x_org_encode, x_drug_encode)
        return org_root_vecs

    def infer(self, org_img, drug_img, greedy=True, perturb_z=True):
        org_root_vecs = self.get_embed(org_img, drug_img)
        root_vecs, root_kl = self.rsample(org_root_vecs, self.R_mean, self.R_var, perturb_z)
        # print(root_vecs.min(), root_vecs.max())
        # try:
        # breakpoint()
        # root_vecs.to("cuda")
        infer_smiles = self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=greedy, max_decode_step=150)
        # except:
        #     breakpoint()
        #     infer_smiles = self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=greedy, max_decode_step=150)
        return infer_smiles, org_root_vecs


    @classmethod
    def from_pretrained(cls,
                        config: dict,
                        ckpt_path_components: tuple,
                        device: str = "cuda",
                        strict: bool = False,
                        key_mapping: dict = None):

        model = cls(config).to(device)
        ckpt_decoder, ckpt_encoder = ckpt_path_components

        def process_keys(checkpoint: dict, prefixes: tuple) -> dict:
            filtered = {}
            for k, v in checkpoint.items():
                k = k.replace("module.", "")
                if key_mapping:
                    for old, new in key_mapping.items():
                        k = k.replace(old, new)
                if k.startswith(prefixes):
                    filtered[k] = v.to(device)
            return filtered

        
        decoder_ckpt = torch.load(ckpt_decoder, map_location=device)
        decoder_dict = process_keys(decoder_ckpt[0], prefixes=('decoder.', 'R_mean.', 'R_var.'))


        encoder_ckpt = torch.load(ckpt_encoder, map_location=device)
        model.encoder.load_state_dict(encoder_ckpt["encoder_state_dict"])
        model.feat.load_state_dict(encoder_ckpt["feat_state_dict"])


        load_info = model.load_state_dict(decoder_dict, strict=strict)

        if load_info.missing_keys:
            print("not find:\n\t" + "\n\t".join(load_info.missing_keys))
        if load_info.unexpected_keys:
            print("other:\n\t" + "\n\t".join(load_info.unexpected_keys))

        return model



class Hierdecoder_classfy_self(nn.Module):

    def __init__(self, config):
        super(Hierdecoder_self, self).__init__()
        vocab = [x.strip("\r\n ").split() for x in open(config["unit_train"]["vocab"])] 
        self.vocab = PairVocab(vocab)
        self.decoder = HierMPNDecoder(self.vocab, common_atom_vocab, config["unit_train"]["rnn_type"], 
                        config["unit_train"]["embed_size"], config["unit_train"]["hidden_size"], config["unit_train"]["latent_size"], 
                        config["unit_train"]["diterT"], config["unit_train"]["diterG"], config["unit_train"]["dropout"])
        self.latent_size = config["unit_train"]["latent_size"]
        self.R_mean = nn.Linear(config["unit_train"]["hidden_size"], self.latent_size)
        self.R_var = nn.Linear(config["unit_train"]["hidden_size"], self.latent_size)

        # molgene_en_feat
        self.encoder = molgene_encoder(config)
        self.feat = molgene_feat(config=config)

        self.classfy_hidden_dim = config["unit_train"]["classfy_hidden_dim"]
        self.fc_classfy_1 = nn.Linear(config["molgene_feat_info"]["decode_embed_dim"], self.classfy_hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc_classfy_2 = nn.Linear(self.classfy_hidden_dim, num_classes)

        # self.embed_model = molgene_en_feat(config)



    def rsample(self, z_vecs, W_mean, W_var, perturb=True):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs( W_var(z_vecs) )
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean).cuda()
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon if perturb else z_mean
        return z_vecs, kl_loss

    def sample(self, batch_size, greedy):
        root_vecs = torch.randn(batch_size, self.latent_size).cuda()
        return self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=greedy, max_decode_step=150)

       
    def forward(self, org_img, drug_img, graphs, tensors, orders, perturb_z=True):  #batches, tensors, all_orders
        tree_tensors, graph_tensors = tensors = make_cuda(tensors)

        # root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)

        # root_vecs = self.embed_model.get_embed(org_img, drug_img)
        x_org_encode = self.encoder(org_img) # (c, h ,w)
        x_drug_encode = self.encoder(drug_img) # (c, h ,w)
        org_root_vecs = self.feat(x_org_encode, x_drug_encode)

        root_vecs, root_kl = self.rsample(org_root_vecs, self.R_mean, self.R_var, perturb_z)
        kl_div = root_kl

        # org_root_vecs
        org_root_vecs_hidden = self.fc_classfy_1(org_root_vecs)
        org_root_vecs_hidden = F.relu(org_root_vecs_hidden) 
        org_root_vecs_hidden = self.dropout(org_root_vecs_hidden)
        logits = self.fc_classfy_2(org_root_vecs_hidden)

        loss, wacc, iacc, tacc, sacc = self.decoder((root_vecs, root_vecs, root_vecs), graphs, tensors, orders)
        return loss, kl_div.item(), wacc, iacc, tacc, sacc, logits

    def get_embed(self, org_img, drug_img):
        x_org_encode = self.encoder(org_img) # (c, h ,w)
        x_drug_encode = self.encoder(drug_img) # (c, h ,w)
        org_root_vecs = self.feat(x_org_encode, x_drug_encode)
        return org_root_vecs

    def infer(self, org_img, drug_img, greedy=True, perturb_z=True):
        org_root_vecs = self.get_embed(org_img, drug_img)
        root_vecs, root_kl = self.rsample(org_root_vecs, self.R_mean, self.R_var, perturb_z)
        print(root_vecs.min(), root_vecs.max())
        # try:
        infer_smiles = self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=greedy, max_decode_step=150)
        # except:
        #     breakpoint()
        #     infer_smiles = self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=greedy, max_decode_step=150)
        return infer_smiles, org_root_vecs


    @classmethod
    def from_pretrained(cls,
                        config: dict,
                        ckpt_path_components: tuple,
                        device: str = "cuda",
                        strict: bool = False,
                        key_mapping: dict = None):

        model = cls(config).to(device)
        ckpt_decoder, ckpt_encoder = ckpt_path_components

        def process_keys(checkpoint: dict, prefixes: tuple) -> dict:
            filtered = {}
            for k, v in checkpoint.items():
                k = k.replace("module.", "")
                if key_mapping:
                    for old, new in key_mapping.items():
                        k = k.replace(old, new)
                if k.startswith(prefixes):
                    filtered[k] = v.to(device)
            return filtered

        
        decoder_ckpt = torch.load(ckpt_decoder, map_location=device)
        decoder_dict = process_keys(decoder_ckpt[0], prefixes=('decoder.', 'R_mean.', 'R_var.'))


        encoder_ckpt = torch.load(ckpt_encoder, map_location=device)
        model.encoder.load_state_dict(encoder_ckpt["encoder_state_dict"])
        model.feat.load_state_dict(encoder_ckpt["feat_state_dict"])


        load_info = model.load_state_dict(decoder_dict, strict=strict)

        if load_info.missing_keys:
            print("not find:\n\t" + "\n\t".join(load_info.missing_keys))
        if load_info.unexpected_keys:
            print("other:\n\t" + "\n\t".join(load_info.unexpected_keys))

        return model
