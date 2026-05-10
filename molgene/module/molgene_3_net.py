import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from module.hgraph2graph.hgraph.mol_graph import MolGraph
from module.hgraph2graph.hgraph.encoder import HierMPNEncoder
from module.hgraph2graph.hgraph.decoder import HierMPNDecoder
from module.hgraph2graph.hgraph.nnutils import *
from module.molgene_net import *
from module.molgene_feat import *
from module.molgene_feat_l1000 import *
from module.hgraph2graph.hgraph import *
from collections import OrderedDict
import torch.nn.functional as F

def make_cuda(tensors):
    tree_tensors, graph_tensors = tensors
    make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x)
    tree_tensors = [make_tensor(x).cuda().long() for x in tree_tensors[:-1]] + [tree_tensors[-1]]
    graph_tensors = [make_tensor(x).cuda().long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    return tree_tensors, graph_tensors


class molgene_3_net(nn.Module):

    def __init__(self, config):
        super(molgene_3_net, self).__init__()
        vocab = [x.strip("\r\n ").split() for x in open(config["unit_3_train"]["vocab"])] 
        self.vocab = PairVocab(vocab)
        self.config = config
        self.hgraph_encoder = HierMPNEncoder(self.vocab, common_atom_vocab, config["unit_train"]["rnn_type"], 
                                            config["unit_train"]["embed_size"], config["unit_train"]["hidden_size"], 
                                            config["unit_train"]["depthT"], config["unit_train"]["depthG"], 
                                            config["unit_train"]["dropout"])
        self.decoder = HierMPNDecoder(self.vocab, common_atom_vocab, config["unit_train"]["rnn_type"], 
                        config["unit_train"]["embed_size"], config["unit_train"]["hidden_size"], config["unit_train"]["latent_size"], 
                        config["unit_train"]["diterT"], config["unit_train"]["diterG"], config["unit_train"]["dropout"])
        self.latent_size = config["unit_train"]["latent_size"]
        self.R_mean = nn.Linear(config["unit_train"]["hidden_size"], self.latent_size)
        self.R_var = nn.Linear(config["unit_train"]["hidden_size"], self.latent_size)
        

        # molgene_en_feat
        self.encoder = molgene_encoder(config)
        self.feat = molgene_feat(config=config)

        self.loss_fn = nn.MSELoss(reduction='mean')


    def rsample(self, z_vecs, W_mean, W_var, perturb=True):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs( W_var(z_vecs) )
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean)  # .cuda()
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon if perturb else z_mean
        return z_vecs, kl_loss

    def sample(self, batch_size, greedy):
        root_vecs = torch.randn(batch_size, self.latent_size).cuda()
        return self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=greedy, max_decode_step=150)

       
    def forward(self, org_img, drug_img, graphs, tensors, orders, beta, perturb_z=True, is_train=True):  #batches, tensors, all_orders
        if is_train: # train all open
            tree_tensors, graph_tensors = tensors = make_cuda(tensors)
            # hgraph_encoder
            hgraph_root_vecs, tree_vecs, _, graph_vecs = self.hgraph_encoder(tree_tensors, graph_tensors)
            # feat_encoder
            x_org_encode = self.encoder(org_img) # (c, h ,w)
            x_drug_encode = self.encoder(drug_img) # (c, h ,w)
            feat_root_vecs = self.feat(x_org_encode, x_drug_encode)
            # rsample(using feat_root_vecs)
            root_vecs, root_kl = self.rsample(feat_root_vecs, self.R_mean, self.R_var, perturb_z)

            ### loss
            # kl_loss
            kl_div = root_kl
            # mse_loss
            mse_loss = self.loss_fn(hgraph_root_vecs, feat_root_vecs)

            mse_loss = mse_loss * 750

            loss_kl_vec = -1
            reconstruct_loss, wacc, iacc, tacc, sacc = self.decoder((root_vecs, root_vecs, root_vecs), graphs, tensors, orders)
            # all
            if self.config["unit_3_train"]["if_mse"]:
                loss = reconstruct_loss + beta * kl_div + mse_loss
            elif self.config["unit_3_train"]["if_vec_kl"]:
                T = self.config["unit_3_train"]["T"]
                teacher_normalized = (hgraph_root_vecs + 1) / 2
                student_normalized = (feat_root_vecs + 1) / 2  
                prob_hgraph_root_vecs = F.softmax(teacher_normalized / T, dim=1)
                prob_feat_root_vecs = F.log_softmax(student_normalized / T, dim=1) 
                loss_kl_vec = F.kl_div(
                    prob_feat_root_vecs,       
                    prob_hgraph_root_vecs,      
                    reduction='batchmean', 
                    log_target=False   
                )
                loss_kl_vec = loss_kl_vec * 2500
                loss = reconstruct_loss + beta * kl_div + mse_loss + loss_kl_vec 
            else:
                loss = reconstruct_loss + beta * kl_div
            # print(reconstruct_loss, "  ", beta * kl_div, "  ", mse_loss)

            return loss, kl_div.item(), wacc, iacc, tacc, sacc, mse_loss, loss_kl_vec
        else:
            tree_tensors, graph_tensors = tensors = make_cuda(tensors)
            # hgraph_encoder
            hgraph_root_vecs, tree_vecs, _, graph_vecs = self.hgraph_encoder(tree_tensors, graph_tensors)
            # rsample(using feat_root_vecs)
            root_vecs, root_kl = self.rsample(hgraph_root_vecs, self.R_mean, self.R_var, perturb_z)

            ### loss
            # kl_loss
            kl_div = root_kl
            reconstruct_loss, wacc, iacc, tacc, sacc = self.decoder((root_vecs, root_vecs, root_vecs), graphs, tensors, orders)
            # all
            loss = reconstruct_loss + beta * kl_div

            return loss, kl_div.item(), wacc, iacc, tacc, sacc, -1, -1


    def get_embed(self, org_img, drug_img):
        x_org_encode = self.encoder(org_img) # (c, h ,w)
        x_drug_encode = self.encoder(drug_img) # (c, h ,w)
        org_root_vecs = self.feat(x_org_encode, x_drug_encode)
        return org_root_vecs


    def infer(self, org_img, drug_img, tensors=None, greedy=True, perturb_z=True, is_mse=True):
        org_root_vecs = self.get_embed(org_img, drug_img)
        root_vecs, root_kl = self.rsample(org_root_vecs, self.R_mean, self.R_var, perturb_z)

        mse_dst = -1
        hgraph_smiles = ["none"]
        if is_mse:
            tree_tensors, graph_tensors = tensors = make_cuda(tensors)
            # hgraph_encoder
            hgraph_root_vecs, tree_vecs, _, graph_vecs = self.hgraph_encoder(tree_tensors, graph_tensors)

            mse_dst = torch.mean((org_root_vecs.clone().detach() - hgraph_root_vecs.clone().detach()) ** 2)

            hgraph_root_vecs, root_kl = self.rsample(hgraph_root_vecs, self.R_mean, self.R_var, perturb_z)

            hgraph_smiles = self.decoder.decode((hgraph_root_vecs, hgraph_root_vecs, hgraph_root_vecs), greedy=greedy, max_decode_step=150)


        infer_smiles = self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=greedy, max_decode_step=150)


        return infer_smiles, hgraph_smiles, org_root_vecs, mse_dst


    @classmethod
    def from_pretrained(cls,
                        config: dict,
                        ckpt_path_components: tuple,
                        device: str = "cuda",
                        strict: bool = False,
                        key_mapping = {"encoder": "hgraph_encoder"}):

        model = cls(config).to(device)
        ckpt_decoder, ckpt_encoder = ckpt_path_components

        def process_keys(checkpoint: dict, prefixes: tuple) -> dict:
            filtered = {}
            for k, v in checkpoint.items():
                k = k.replace("module.", "")
                if key_mapping:
                    for old, new in key_mapping.items():
                        if k.startswith(old):
                            k = new + k[len(old):]
                if k.startswith(prefixes):
                    filtered[k] = v.to(device)
            return filtered

        decoder_ckpt = torch.load(ckpt_decoder, map_location=device)
        decoder_dict = process_keys(decoder_ckpt[0], prefixes=('decoder.', 'R_mean.', 'R_var.','hgraph_encoder.'))


        encoder_ckpt = torch.load(ckpt_encoder, map_location=device)
        model.encoder.load_state_dict(encoder_ckpt["encoder_state_dict"])
        model.feat.load_state_dict(encoder_ckpt["feat_state_dict"])


        load_info = model.load_state_dict(decoder_dict, strict=strict)

        print(f"success: {len(load_info.missing_keys)} missing, "
              f"{len(load_info.unexpected_keys)} unexpected")

        if load_info.missing_keys:
            print("not find:\n\t" + "\n\t".join(load_info.missing_keys))
        if load_info.unexpected_keys:
            print("other:\n\t" + "\n\t".join(load_info.unexpected_keys))

        beta = decoder_ckpt[3]

        return model, beta





class molgene_3_kl_net(nn.Module):

    def __init__(self, config):
        super(molgene_3_kl_net, self).__init__()
        vocab = [x.strip("\r\n ").split() for x in open(config["unit_3_train"]["vocab"])] 
        self.vocab = PairVocab(vocab)
        self.config = config
        self.hgraph_encoder = HierMPNEncoder(self.vocab, common_atom_vocab, config["unit_train"]["rnn_type"], 
                                            config["unit_train"]["embed_size"], config["unit_train"]["hidden_size"], 
                                            config["unit_train"]["depthT"], config["unit_train"]["depthG"], 
                                            config["unit_train"]["dropout"])
        self.decoder = HierMPNDecoder(self.vocab, common_atom_vocab, config["unit_train"]["rnn_type"], 
                        config["unit_train"]["embed_size"], config["unit_train"]["hidden_size"], config["unit_train"]["latent_size"], 
                        config["unit_train"]["diterT"], config["unit_train"]["diterG"], config["unit_train"]["dropout"])
        self.latent_size = config["unit_train"]["latent_size"]
        self.R_mean = nn.Linear(config["unit_train"]["hidden_size"], self.latent_size)
        self.R_var = nn.Linear(config["unit_train"]["hidden_size"], self.latent_size)

        self.feat_R_mean = nn.Linear(config["unit_train"]["hidden_size"], self.latent_size)
        self.feat_R_var = nn.Linear(config["unit_train"]["hidden_size"], self.latent_size)
        

        # molgene_en_feat
        self.encoder = molgene_encoder(config)
        self.feat = molgene_feat(config=config)

        self.loss_fn = nn.MSELoss(reduction='mean')



    def kl_loss_gaussian(self, mu1, log_var1, mu2, log_var2):

        var1 = torch.exp(log_var1)
        var2 = torch.exp(log_var2)
        
        kl = 0.5 * (
            log_var2 - log_var1 +                       # log(σ2²/σ1²) + (σ1² + (μ1-μ2)²)/σ2² - 1
            (var1 + (mu1 - mu2).pow(2)) / var2 - 1       # 
        )
        
        kl_loss = torch.sum(kl, dim=1).mean()
        return kl_loss

    def rsample(self, z_vecs, W_mean, W_var, perturb=True):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs( W_var(z_vecs) )
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean)  # .cuda()
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon if perturb else z_mean
        return z_vecs, kl_loss

    def rsample_train(self, z_vecs, W_mean, W_var, perturb=True):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs( W_var(z_vecs) )
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean)  # .cuda()
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon if perturb else z_mean
        return z_vecs, kl_loss, z_mean, z_log_var

    def sample(self, batch_size, greedy):
        root_vecs = torch.randn(batch_size, self.latent_size).cuda()
        return self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=greedy, max_decode_step=150)

       
    def forward(self, org_img, drug_img, graphs, tensors, orders, beta, perturb_z=True, is_train=True):  #batches, tensors, all_orders
        if is_train: # train all open
            tree_tensors, graph_tensors = tensors = make_cuda(tensors)
            # hgraph_encoder
            hgraph_root_vecs, tree_vecs, _, graph_vecs = self.hgraph_encoder(tree_tensors, graph_tensors)
            # feat_encoder
            x_org_encode = self.encoder(org_img) # (c, h ,w)
            x_drug_encode = self.encoder(drug_img) # (c, h ,w)
            feat_root_vecs = self.feat(x_org_encode, x_drug_encode)
            # rsample(using feat_root_vecs)
            root_vecs, root_kl, feat_mean, feat_log_var = self.rsample_train(feat_root_vecs, self.feat_R_mean, self.feat_R_var, perturb_z)

            _, _, hgraph_mean, hgraph_log_var = self.rsample_train(hgraph_root_vecs, self.R_mean, self.R_var, perturb_z)

            ### loss
            # kl_loss
            kl_div = root_kl

            ### mse_loss
            # mean_loss
            mean_mse_loss = self.loss_fn(hgraph_mean, feat_mean)
            mean_mse_loss = mean_mse_loss * 300
            # var_loss
            var_mse_loss = self.loss_fn(hgraph_log_var, feat_log_var)
            var_mse_loss = var_mse_loss * 300

            ### feat_net hgraph kl loss
            feat_hgraph_kl_loss = self.kl_loss_gaussian(feat_mean, feat_log_var, hgraph_mean, hgraph_log_var)




            loss_kl_vec = -1
            reconstruct_loss, wacc, iacc, tacc, sacc = self.decoder((root_vecs, root_vecs, root_vecs), graphs, tensors, orders)

            if self.config["unit_3_train"]["molgene_3_kl_net"]["org_kl_loss"]:
                loss = reconstruct_loss + mean_mse_loss + var_mse_loss # + feat_hgraph_kl_loss  + beta * kl_div
            else:
                loss = reconstruct_loss + mean_mse_loss + var_mse_loss # + feat_hgraph_kl_loss
            # print(reconstruct_loss, "  ", beta * kl_div, "  ", mse_loss)

            return loss, kl_div.item(), wacc, iacc, tacc, sacc, mean_mse_loss, var_mse_loss, feat_hgraph_kl_loss, reconstruct_loss
        else:
            tree_tensors, graph_tensors = tensors = make_cuda(tensors)
            # hgraph_encoder
            hgraph_root_vecs, tree_vecs, _, graph_vecs = self.hgraph_encoder(tree_tensors, graph_tensors)
            # rsample(using feat_root_vecs)
            root_vecs, root_kl, hgraph_mean, hgraph_var = self.rsample_train(hgraph_root_vecs, self.R_mean, self.R_var, perturb_z)
            # z_vecs, kl_loss, z_mean, z_log_var

            ### loss
            # kl_loss
            kl_div = root_kl
            reconstruct_loss, wacc, iacc, tacc, sacc = self.decoder((root_vecs, root_vecs, root_vecs), graphs, tensors, orders)
            # all
            loss = reconstruct_loss + beta * kl_div
            # print(reconstruct_loss, "  ", beta * kl_div)

            return loss, kl_div.item(), wacc, iacc, tacc, sacc, -1, -1, -1, reconstruct_loss


    def get_embed(self, org_img, drug_img):
        x_org_encode = self.encoder(org_img) # (c, h ,w)
        x_drug_encode = self.encoder(drug_img) # (c, h ,w)
        org_root_vecs = self.feat(x_org_encode, x_drug_encode)
        return org_root_vecs


    def infer(self, org_img, drug_img, tensors=None, greedy=True, perturb_z=True, is_mse=True):
        org_root_vecs = self.get_embed(org_img, drug_img)
        root_vecs, root_kl = self.rsample(org_root_vecs, self.feat_R_mean, self.feat_R_var, perturb_z)

        mse_dst = -1
        if is_mse:
            tree_tensors, graph_tensors = tensors = make_cuda(tensors)
            # hgraph_encoder
            hgraph_root_vecs, tree_vecs, _, graph_vecs = self.hgraph_encoder(tree_tensors, graph_tensors)

            mse_dst = torch.mean((org_root_vecs.clone().detach() - hgraph_root_vecs.clone().detach()) ** 2)

            hgraph_root_vecs, root_kl = self.rsample(hgraph_root_vecs, self.R_mean, self.R_var, perturb_z)

        infer_smiles = self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=greedy, max_decode_step=150)

        hgraph_smiles = self.decoder.decode((hgraph_root_vecs, hgraph_root_vecs, hgraph_root_vecs), greedy=greedy, max_decode_step=150)

        return infer_smiles, hgraph_smiles, org_root_vecs, mse_dst


    @classmethod
    def from_pretrained(cls,
                        config: dict,
                        ckpt_path_components: tuple,
                        device: str = "cuda",
                        strict: bool = False,
                        key_mapping_input = {"encoder": "hgraph_encoder"}):

        model = cls(config).to(device)
        ckpt_decoder, ckpt_encoder = ckpt_path_components

        def process_keys(checkpoint: dict, prefixes: tuple, key_mapping_input) -> dict:
            filtered = {}
            for k, v in checkpoint.items():
                k = k.replace("module.", "")
                if key_mapping_input:
                    for old, new in key_mapping_input.items():
                        if k.startswith(old):
                            k = new + k[len(old):]
                if k.startswith(prefixes):
                    filtered[k] = v.to(device)
            return filtered

        
        decoder_ckpt = torch.load(ckpt_decoder, map_location=device)
        decoder_dict = process_keys(decoder_ckpt[0], prefixes=('decoder.', 'R_mean.', 'R_var.','hgraph_encoder.'),key_mapping_input={"encoder": "hgraph_encoder"})
        mean_var_dict = process_keys(decoder_ckpt[0], prefixes=('feat_R_mean.', 'feat_R_var.'),key_mapping_input={"R_mean": "feat_R_mean", "R_var": "feat_R_var"})

        encoder_ckpt = torch.load(ckpt_encoder, map_location=device)
        model.encoder.load_state_dict(encoder_ckpt["encoder_state_dict"])
        model.feat.load_state_dict(encoder_ckpt["feat_state_dict"])

        mean_var_load_info = model.load_state_dict(mean_var_dict, strict=strict)
        if mean_var_load_info.missing_keys:
            print("not find:\n\t" + "\n\t".join(mean_var_load_info.missing_keys))
        if mean_var_load_info.unexpected_keys:
            print("other:\n\t" + "\n\t".join(mean_var_load_info.unexpected_keys))

        load_info = model.load_state_dict(decoder_dict, strict=strict)

        print(f"success: {len(load_info.missing_keys)} missing, "
              f"{len(load_info.unexpected_keys)} unexpected")

        if load_info.missing_keys:
            print("not find:\n\t" + "\n\t".join(load_info.missing_keys))
        if load_info.unexpected_keys:
            print("other:\n\t" + "\n\t".join(load_info.unexpected_keys))

        beta = decoder_ckpt[3]

        return model, beta

class molgene_3_kl_l1000_net(nn.Module):

    def __init__(self, config):
        super(molgene_3_kl_l1000_net, self).__init__()
        vocab = [x.strip("\r\n ").split() for x in open(config["unit_3_train"]["vocab"])] 
        self.vocab = PairVocab(vocab)
        self.config = config
        self.encoder = HierMPNEncoder(self.vocab, common_atom_vocab, config["unit_train"]["rnn_type"], 
                                            config["unit_train"]["embed_size"], config["unit_train"]["hidden_size"], 
                                            config["unit_train"]["depthT"], config["unit_train"]["depthG"], 
                                            config["unit_train"]["dropout"])
        self.decoder = HierMPNDecoder(self.vocab, common_atom_vocab, config["unit_train"]["rnn_type"], 
                        config["unit_train"]["embed_size"], config["unit_train"]["hidden_size"], config["unit_train"]["latent_size"], 
                        config["unit_train"]["diterT"], config["unit_train"]["diterG"], config["unit_train"]["dropout"])
        self.latent_size = config["unit_train"]["latent_size"]
        self.R_mean = nn.Linear(config["unit_train"]["hidden_size"], self.latent_size)
        self.R_var = nn.Linear(config["unit_train"]["hidden_size"], self.latent_size)

        self.feat_R_mean = nn.Linear(config["unit_train"]["hidden_size"], self.latent_size)
        self.feat_R_var = nn.Linear(config["unit_train"]["hidden_size"], self.latent_size)
        
        self.feat = molgene_feat_1000(input_len=978, embed_dim=16, num_heads=4, decode_embed_dim=250)

        self.loss_fn = nn.MSELoss(reduction='mean')



    def kl_loss_gaussian(self, mu1, log_var1, mu2, log_var2):

        var1 = torch.exp(log_var1)
        var2 = torch.exp(log_var2)
        
        kl = 0.5 * (
            log_var2 - log_var1 +                       # log(σ2²/σ1²) + (σ1² + (μ1-μ2)²)/σ2² - 1
            (var1 + (mu1 - mu2).pow(2)) / var2 - 1       # 
        )
        
        kl_loss = torch.sum(kl, dim=1).mean()
        return kl_loss

    def rsample(self, z_vecs, W_mean, W_var, perturb=True):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs( W_var(z_vecs) )
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean)  # .cuda()
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon if perturb else z_mean
        return z_vecs, kl_loss

    def rsample_train(self, z_vecs, W_mean, W_var, perturb=True):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs( W_var(z_vecs) )
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean)  # .cuda()
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon if perturb else z_mean
        return z_vecs, kl_loss, z_mean, z_log_var

    def sample(self, batch_size, greedy):
        root_vecs = torch.randn(batch_size, self.latent_size).cuda()
        return self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=greedy, max_decode_step=150)

       
    def forward(self, org_img, drug_img, graphs, tensors, orders, beta, perturb_z=True, is_train=True):  #batches, tensors, all_orders
        if is_train: # train all open
            tree_tensors, graph_tensors = tensors = make_cuda(tensors)
            # hgraph_encoder
            hgraph_root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)
            # feat_encoder
            # x_org_encode = self.encoder(org_img) # (c, h ,w)
            # x_drug_encode = self.encoder(drug_img) # (c, h ,w)
            feat_root_vecs = self.feat(org_img, drug_img)
            # rsample(using feat_root_vecs)
            root_vecs, root_kl, feat_mean, feat_log_var = self.rsample_train(feat_root_vecs, self.feat_R_mean, self.feat_R_var, perturb_z)

            _, _, hgraph_mean, hgraph_log_var = self.rsample_train(hgraph_root_vecs, self.R_mean, self.R_var, perturb_z)

            ### loss
            # kl_loss
            kl_div = root_kl

            ### mse_loss
            # mean_loss
            mean_mse_loss = self.loss_fn(hgraph_mean, feat_mean)
            mean_mse_loss = mean_mse_loss * 300
            # var_loss
            var_mse_loss = self.loss_fn(hgraph_log_var, feat_log_var)
            var_mse_loss = var_mse_loss * 300

            ### feat_net hgraph kl loss
            feat_hgraph_kl_loss = self.kl_loss_gaussian(feat_mean, feat_log_var, hgraph_mean, hgraph_log_var)




            loss_kl_vec = -1
            reconstruct_loss, wacc, iacc, tacc, sacc = self.decoder((root_vecs, root_vecs, root_vecs), graphs, tensors, orders)

            if self.config["unit_3_train"]["molgene_3_kl_net"]["org_kl_loss"]:
                loss = reconstruct_loss + mean_mse_loss + var_mse_loss 
            else:
                loss = reconstruct_loss + mean_mse_loss + var_mse_loss 

            return loss, kl_div.item(), wacc, iacc, tacc, sacc, mean_mse_loss, var_mse_loss, feat_hgraph_kl_loss, reconstruct_loss
        else:
            tree_tensors, graph_tensors = tensors = make_cuda(tensors)
            # hgraph_encoder
            hgraph_root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)
            # rsample(using feat_root_vecs)
            root_vecs, root_kl, hgraph_mean, hgraph_var = self.rsample_train(hgraph_root_vecs, self.R_mean, self.R_var, perturb_z)
            # z_vecs, kl_loss, z_mean, z_log_var

            ### loss
            # kl_loss
            kl_div = root_kl
            reconstruct_loss, wacc, iacc, tacc, sacc = self.decoder((root_vecs, root_vecs, root_vecs), graphs, tensors, orders)
            # all
            loss = reconstruct_loss + beta * kl_div

            return loss, kl_div.item(), wacc, iacc, tacc, sacc, -1, -1, -1, reconstruct_loss


    def get_embed(self, org_img, drug_img):
        org_root_vecs = self.feat(org_img, drug_img)
        return org_root_vecs
    
    def get_embed_hgraph(self, tensors):
        tree_tensors, graph_tensors = tensors = make_cuda(tensors)
        hgraph_root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)
        return hgraph_root_vecs
    
    def reconstruct(self, tensors):
        tree_tensors, graph_tensors = tensors = make_cuda(tensors)
        root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)

        root_vecs, root_kl = self.rsample(root_vecs, self.R_mean, self.R_var, perturb=False)
        return self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=True, max_decode_step=150)


    def infer(self, org_img, drug_img, tensors=None, greedy=True, perturb_z=True, is_mse=True):
        org_root_vecs = self.get_embed(org_img, drug_img)
        root_vecs, root_kl = self.rsample(org_root_vecs, self.feat_R_mean, self.feat_R_var, perturb_z)

        mse_dst = -1
        if is_mse:
            tree_tensors, graph_tensors = tensors = make_cuda(tensors)
            hgraph_root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)

            mse_dst = torch.mean((org_root_vecs.clone().detach() - hgraph_root_vecs.clone().detach()) ** 2)

            hgraph_root_vecs, root_kl = self.rsample(hgraph_root_vecs, self.R_mean, self.R_var, perturb_z)


        infer_smiles = self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=greedy, max_decode_step=150)

        hgraph_smiles = self.decoder.decode((hgraph_root_vecs, hgraph_root_vecs, hgraph_root_vecs), greedy=greedy, max_decode_step=150)
        return infer_smiles, hgraph_smiles, org_root_vecs, mse_dst


    @classmethod
    def from_pretrained(cls,
                        config: dict,
                        ckpt_path_components: tuple,
                        device: str = "cuda",
                        strict: bool = False,
                        key_mapping_input = {"encoder": "hgraph_encoder"}):
        model = cls(config).to(device)
        ckpt_decoder, ckpt_encoder = ckpt_path_components

        def process_keys(checkpoint: dict, prefixes: tuple, key_mapping_input) -> dict:
            filtered = {}
            for k, v in checkpoint.items():
                k = k.replace("module.", "")
                if key_mapping_input:
                    for old, new in key_mapping_input.items():
                        if k.startswith(old):
                            k = new + k[len(old):]
                if k.startswith(prefixes):
                    filtered[k] = v.to(device)
            return filtered
        
        decoder_ckpt = torch.load(ckpt_decoder, map_location=device)
        decoder_dict = process_keys(decoder_ckpt[0], prefixes=('decoder.', 'R_mean.', 'R_var.','hgraph_encoder.'),key_mapping_input={"encoder": "hgraph_encoder"})
        mean_var_dict = process_keys(decoder_ckpt[0], prefixes=('feat_R_mean.', 'feat_R_var.'),key_mapping_input={"R_mean": "feat_R_mean", "R_var": "feat_R_var"})

        encoder_ckpt = torch.load(ckpt_encoder, map_location=device)
        model.encoder.load_state_dict(encoder_ckpt["encoder_state_dict"])
        model.feat.load_state_dict(encoder_ckpt["feat_state_dict"])

        mean_var_load_info = model.load_state_dict(mean_var_dict, strict=strict)
        if mean_var_load_info.missing_keys:
            print("not find:\n\t" + "\n\t".join(mean_var_load_info.missing_keys))
        if mean_var_load_info.unexpected_keys:
            print("other:\n\t" + "\n\t".join(mean_var_load_info.unexpected_keys))

        load_info = model.load_state_dict(decoder_dict, strict=strict)

        if load_info.missing_keys:
            print("not find:\n\t" + "\n\t".join(load_info.missing_keys))
        if load_info.unexpected_keys:
            print("other:\n\t" + "\n\t".join(load_info.unexpected_keys))

        beta = decoder_ckpt[3]

        return model, beta


class molgene_3_frozen_dropout_net(nn.Module):

    def __init__(self, config):
        super(molgene_3_frozen_dropout_net, self).__init__()
        vocab = [x.strip("\r\n ").split() for x in open(config["unit_3_train"]["vocab"])] 
        self.vocab = PairVocab(vocab)
        self.config = config
        self.hgraph_encoder = HierMPNEncoder(self.vocab, common_atom_vocab, config["unit_train"]["rnn_type"], 
                                            config["unit_train"]["embed_size"], config["unit_train"]["hidden_size"], 
                                            config["unit_train"]["depthT"], config["unit_train"]["depthG"], 
                                            config["unit_train"]["dropout"])
        self.decoder = HierMPNDecoder(self.vocab, common_atom_vocab, config["unit_train"]["rnn_type"], 
                        config["unit_train"]["embed_size"], config["unit_train"]["hidden_size"], config["unit_train"]["latent_size"], 
                        config["unit_train"]["diterT"], config["unit_train"]["diterG"], config["unit_train"]["dropout"])
        self.latent_size = config["unit_train"]["latent_size"]
        self.R_mean = nn.Linear(config["unit_train"]["hidden_size"], self.latent_size)
        self.R_var = nn.Linear(config["unit_train"]["hidden_size"], self.latent_size)

        self.feat_R_mean = nn.Linear(config["unit_train"]["hidden_size"], self.latent_size)
        self.feat_R_var = nn.Linear(config["unit_train"]["hidden_size"], self.latent_size)
        

        # molgene_en_feat
        self.encoder = molgene_encoder(config)
        # self.feat = molgene_feat(config=config)
        self.feat = molgene_feat_simple(b=config["unit_3_train"]["batch_size"])

        self.loss_fn = nn.MSELoss(reduction='mean')

        # self.embed_model = molgene_en_feat(config)


    def kl_loss_gaussian(self, mu1, log_var1, mu2, log_var2):

        var1 = torch.exp(log_var1)
        var2 = torch.exp(log_var2)
        
        kl = 0.5 * (
            log_var2 - log_var1 +                       # log(σ2²/σ1²) + (σ1² + (μ1-μ2)²)/σ2² - 1
            (var1 + (mu1 - mu2).pow(2)) / var2 - 1       # 
        )
        
        kl_loss = torch.sum(kl, dim=1).mean()
        return kl_loss

    def rsample(self, z_vecs, W_mean, W_var, perturb=True):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs( W_var(z_vecs) )
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean)  # .cuda()
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon if perturb else z_mean
        return z_vecs, kl_loss

    def rsample_train(self, z_vecs, W_mean, W_var, perturb=True):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs( W_var(z_vecs) )
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean)  # .cuda()
        # breakpoint()
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon if perturb else z_mean
        return z_vecs, kl_loss, z_mean, z_log_var

    def sample(self, batch_size, greedy):
        root_vecs = torch.randn(batch_size, self.latent_size).cuda()
        return self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=greedy, max_decode_step=150)

       
    def forward(self, org_img, drug_img, graphs, tensors, orders, beta, perturb_z=True, is_train=True):  #batches, tensors, all_orders
        if is_train: # train all open
            tree_tensors, graph_tensors = tensors = make_cuda(tensors)
            # hgraph_encoder
            hgraph_root_vecs, tree_vecs, _, graph_vecs = self.hgraph_encoder(tree_tensors, graph_tensors)
            # feat_encoder
            x_org_encode = self.encoder(org_img) # (c, h ,w)
            x_drug_encode = self.encoder(drug_img) # (c, h ,w)
            feat_root_vecs = self.feat(x_org_encode, x_drug_encode)
            # rsample(using feat_root_vecs)
            root_vecs, root_kl, feat_mean, feat_log_var = self.rsample_train(feat_root_vecs, self.feat_R_mean, self.feat_R_var, perturb_z)

            _, _, hgraph_mean, hgraph_log_var = self.rsample_train(hgraph_root_vecs, self.R_mean, self.R_var, perturb_z)

            ### loss
            # kl_loss
            kl_div = root_kl

            ### mse_loss
            # mean_loss
            mean_mse_loss = self.loss_fn(hgraph_mean, feat_mean)
            mean_mse_loss = mean_mse_loss * 300
            # var_loss
            var_mse_loss = self.loss_fn(hgraph_log_var, feat_log_var)
            var_mse_loss = var_mse_loss * 300

            ### feat_net hgraph kl loss
            feat_hgraph_kl_loss = self.kl_loss_gaussian(feat_mean, feat_log_var, hgraph_mean, hgraph_log_var)




            loss_kl_vec = -1
            reconstruct_loss, wacc, iacc, tacc, sacc = self.decoder((root_vecs, root_vecs, root_vecs), graphs, tensors, orders)
            if self.config["unit_3_train"]["molgene_3_kl_net"]["org_kl_loss"]:
                loss = reconstruct_loss + mean_mse_loss + var_mse_loss + feat_hgraph_kl_loss 
            else:
                loss = reconstruct_loss + mean_mse_loss + var_mse_loss 

            return loss, kl_div.item(), wacc, iacc, tacc, sacc, mean_mse_loss, var_mse_loss, feat_hgraph_kl_loss, reconstruct_loss
        else:
            tree_tensors, graph_tensors = tensors = make_cuda(tensors)
            # hgraph_encoder
            hgraph_root_vecs, tree_vecs, _, graph_vecs = self.hgraph_encoder(tree_tensors, graph_tensors)
            # rsample(using feat_root_vecs)
            root_vecs, root_kl, hgraph_mean, hgraph_var = self.rsample_train(hgraph_root_vecs, self.R_mean, self.R_var, perturb_z)
            # z_vecs, kl_loss, z_mean, z_log_var

            ### loss
            # kl_loss
            kl_div = root_kl
            reconstruct_loss, wacc, iacc, tacc, sacc = self.decoder((root_vecs, root_vecs, root_vecs), graphs, tensors, orders)
            # all
            loss = reconstruct_loss + beta * kl_div
            # print(reconstruct_loss, "  ", beta * kl_div)

            return loss, kl_div.item(), wacc, iacc, tacc, sacc, -1, -1, -1, reconstruct_loss


    def get_embed(self, org_img, drug_img):
        x_org_encode = self.encoder(org_img) # (c, h ,w)
        x_drug_encode = self.encoder(drug_img) # (c, h ,w)
        org_root_vecs = self.feat(x_org_encode, x_drug_encode)
        return org_root_vecs


    def infer(self, org_img, drug_img, tensors=None, greedy=True, perturb_z=True, is_mse=True):
        org_root_vecs = self.get_embed(org_img, drug_img)
        root_vecs, root_kl = self.rsample(org_root_vecs, self.feat_R_mean, self.feat_R_var, perturb_z)

        mse_dst = -1
        if is_mse:
            tree_tensors, graph_tensors = tensors = make_cuda(tensors)
            # hgraph_encoder
            hgraph_root_vecs, tree_vecs, _, graph_vecs = self.hgraph_encoder(tree_tensors, graph_tensors)

            mse_dst = torch.mean((org_root_vecs.clone().detach() - hgraph_root_vecs.clone().detach()) ** 2)

            hgraph_root_vecs, root_kl = self.rsample(hgraph_root_vecs, self.R_mean, self.R_var, perturb_z)



        infer_smiles = self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=greedy, max_decode_step=150)

        hgraph_smiles = self.decoder.decode((hgraph_root_vecs, hgraph_root_vecs, hgraph_root_vecs), greedy=greedy, max_decode_step=150)
        return infer_smiles, hgraph_smiles, org_root_vecs, mse_dst


    @classmethod
    def from_pretrained(cls,
                        config: dict,
                        ckpt_path_components: tuple,
                        device: str = "cuda",
                        strict: bool = False,
                        key_mapping_input = {"encoder": "hgraph_encoder"}):

        model = cls(config).to(device)
        ckpt_decoder, ckpt_encoder = ckpt_path_components

        def process_keys(checkpoint: dict, prefixes: tuple, key_mapping_input) -> dict:
            filtered = {}
            for k, v in checkpoint.items():
                k = k.replace("module.", "")
                if key_mapping_input:
                    for old, new in key_mapping_input.items():
                        if k.startswith(old):
                            k = new + k[len(old):]
                if k.startswith(prefixes):
                    filtered[k] = v.to(device)
            return filtered

        
        decoder_ckpt = torch.load(ckpt_decoder, map_location=device)
        decoder_dict = process_keys(decoder_ckpt[0], prefixes=('decoder.', 'R_mean.', 'R_var.','hgraph_encoder.'),key_mapping_input={"encoder": "hgraph_encoder"})
        mean_var_dict = process_keys(decoder_ckpt[0], prefixes=('feat_R_mean.', 'feat_R_var.'),key_mapping_input={"R_mean": "feat_R_mean", "R_var": "feat_R_var"})

        encoder_ckpt = torch.load(ckpt_encoder, map_location=device)
        model.encoder.load_state_dict(encoder_ckpt["encoder_state_dict"])

        mean_var_load_info = model.load_state_dict(mean_var_dict, strict=strict)
        if mean_var_load_info.missing_keys:
            print("not find:\n\t" + "\n\t".join(mean_var_load_info.missing_keys))
        if mean_var_load_info.unexpected_keys:
            print("other:\n\t" + "\n\t".join(mean_var_load_info.unexpected_keys))

        load_info = model.load_state_dict(decoder_dict, strict=strict)

        if load_info.missing_keys:
            print("not find:\n\t" + "\n\t".join(load_info.missing_keys))
        if load_info.unexpected_keys:
            print("other:\n\t" + "\n\t".join(load_info.unexpected_keys))

        beta = decoder_ckpt[3]

        return model, beta


