import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad

from scipy import sparse
from scipy.stats import wasserstein_distance
from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import preprocessing
from tqdm import tqdm


def Condition_encoder(condition_list):
    """
    Encode conditions of Annotated `adata` matrix with one-hot.
    """

    values = np.array(condition_list)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
              
    return onehot_encoded         

def Drug_dose_encoder(drug_SMILES_list: list, dose_list: list, num_Bits=1024, comb_num=1):
    """
    Encode SMILES of drug to rFCFP fingerprint
    """
    drug_len = len(drug_SMILES_list)
    fcfp4_array = np.zeros((drug_len, num_Bits))

    if comb_num==1:
        for i, smiles in enumerate(drug_SMILES_list):
            smi = smiles
            mol = Chem.MolFromSmiles(smi)
            fcfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=num_Bits).ToBitString()
            fcfp4_list = np.array(list(fcfp4), dtype=np.float32)
            fcfp4_list = fcfp4_list*np.log10(dose_list[i]+1)
            fcfp4_array[i] = fcfp4_list
    else:
        for i, smiles in enumerate(drug_SMILES_list):
            smiles_list = smiles.split('+')
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                fcfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=num_Bits).ToBitString()
                fcfp4_list = np.array(list(fcfp4), dtype=np.float32)
                fcfp4_list = fcfp4_list*np.log10(float(dose_list[i])+1)
                fcfp4_array[i] += fcfp4_list
    return fcfp4_array 

def Drug_SMILES_encode(drug_SMILES_list: list, num_Bits=1024):
    """
    Encode SMILES of drug to FCFP fingerprint
    """
    drug_len = len(drug_SMILES_list)
    fcfp4_array = np.zeros((drug_len, num_Bits))

    for i, smiles in enumerate(drug_SMILES_list):
        smi = smiles
        mol = Chem.MolFromSmiles(smi)
        fcfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=num_Bits).ToBitString()
        fcfp4_list = np.array(list(fcfp4))
        fcfp4_array[i] = fcfp4_list

    return fcfp4_array


def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma


def rank_genes_groups_by_cov(
    adata,
    groupby,
    control_group,
    covariate,
    pool_doses=False,
    n_genes=2,
    rankby_abs=True,
    key_added='rank_genes_groups_cov',
    return_dict=False,
):

    gene_dict = {}
    cov_categories = adata.obs[covariate].unique()
    for cov_cat in cov_categories:
        #name of the control group in the groupby obs column
        control_group_cov = '_'.join([cov_cat, control_group])

        #subset adata to cells belonging to a covariate category
        adata_cov = adata[adata.obs[covariate]==cov_cat]

        #compute DEGs
        sc.tl.rank_genes_groups(
            adata_cov,
            groupby=groupby,
            reference=control_group_cov,
            rankby_abs=rankby_abs,
            n_genes=n_genes,
            use_raw=False
        )

        #add entries to dictionary of gene sets
        de_genes = pd.DataFrame(adata_cov.uns['rank_genes_groups']['names'])
        for group in de_genes:
            gene_dict[group] = de_genes[group].tolist()

    adata.uns[key_added] = gene_dict

    if return_dict:
        return gene_dict




def rank_genes_groups_by_drug(
    adata,
    groupby,
    control_group,
    pool_doses=False,
    n_genes=2,
    rankby_abs=True,
    key_added='rank_genes_groups_drug',
    return_dict=False,
):

    gene_dict = {}
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        reference=control_group,
        rankby_abs=rankby_abs,
        n_genes=n_genes,
        use_raw=False
    )

    #add entries to dictionary of gene sets
    de_genes = pd.DataFrame(adata.uns['rank_genes_groups']['names'])
    for group in de_genes:
        gene_dict[group] = de_genes[group].tolist()

    adata.uns[key_added] = gene_dict

    if return_dict:
        return gene_dict




def pearson_mean(data1, data2):
    sum_pearson_1 = 0
    sum_pearson_2 = 0
    for i in range(data1.shape[0]):
        pearsonr_ = pearsonr(data1[i], data2[i])
        sum_pearson_1 += pearsonr_[0]
        sum_pearson_2 += pearsonr_[1]
    return sum_pearson_1/data1.shape[0], sum_pearson_2/data1.shape[0]

def pearson_list(data1, data2):
    pearson_list = np.zeros(data1.shape[0])
    for i in range(data1.shape[0]):
        pearsonr_ = pearsonr(data1[i], data2[i])
        pearson_list[i] = pearsonr_[0]
    return pearson_list


def r2_mean(data1, data2):
    sum_r2_1 = 0
    for i in range(data1.shape[0]):
        r2_score_ = r2_score(data1[i], data2[i])
        sum_r2_1 += r2_score_           
    return sum_r2_1/data1.shape[0]

def mse_mean(data1, data2):
    sum_mse_1 = 0
    for i in range(data1.shape[0]):
        mse_score_ = mean_squared_error(data1[i], data2[i])
        sum_mse_1 += mse_score_           
    return sum_mse_1/data1.shape[0]

def z_score(control_array, drug_array):
    scaler = preprocessing.StandardScaler()
    array_all = np.concatenate((control_array, drug_array),axis=0)
    scaler.fit(array_all)
    control_array_z = scaler.transform(control_array)
    drug_array_z = scaler.transform(drug_array)
    return control_array_z, drug_array_z

def contribution_df(data):
    data['cov_drug_name'] = data.index
    data['cell_type'] = data.cov_drug_name.apply(lambda x: str(x).split('_')[0])
    data['condition'] = data.cov_drug_name.apply(lambda x: '_'.join(str(x).split('_')[1:]))
    return data



def condition_fc_groups_by_cov(
    data,
    groupby,
    control_group,
    covariate
):

    condition_exp_mean = {}
    control_exp_mean = {}
    fold_change = {}

    cov_categories = data[covariate].unique()
    for cov_cat in cov_categories:
        #name of the control group in the groupby obs column
        control_group_cov = '_'.join([cov_cat, control_group])

        #subset adata to cells belonging to a covariate category
        adata_cov_df = data[data[covariate]==cov_cat]
        adata_cov_df['condition'] = data[groupby]


        control_mean = adata_cov_df[adata_cov_df.cov_drug_name == control_group_cov].mean(numeric_only=True)+1e-6
        control_exp_mean[control_group_cov] = control_mean

        for cond, df in tqdm(adata_cov_df.groupby('condition')): 
            if df.shape[0] != 0 :
                if cond != control_group_cov:
                    drug_mean = df.mean(numeric_only=True)+1e-6
                    fold_change[cond] = drug_mean/control_mean
                    condition_exp_mean[cond] = drug_mean

    return condition_exp_mean, control_exp_mean, fold_change




def computecs(qup, qdown, expression):
    '''
    This function takes qup & qdown, which are lists of gene
    names, and  expression, a panda data frame of the expressions
    of genes as input, and output the connectivity score vector
    '''
    r1 = ranklist(expression)
    if qup and qdown:
        esup = computees(qup, r1)
        esdown = computees(qdown, r1)
        w = []
        for i in range(len(esup)):
            if esup[i]*esdown[i] <= 0:
                w.append(esup[i]-esdown[i])
            else:
                w.append(0)
        return pd.DataFrame(w, expression.columns)
    elif qup and qdown==None:
        esup = computees(qup, r1)
        return pd.DataFrame(esup, expression.columns)
    elif qup == None and qdown:
        esdown = computees(qdown, r1)
        return pd.DataFrame(esdown, expression.columns)
    else:
        return None

def computees(q, r1):
    '''
    This function takes q, a list of gene names, and r1, a panda data
    frame as the input, and output the enrichment score vector
    '''
    if len(q) == 0:
        ks = 0
    elif len(q) == 1:
        ks = r1.loc[q,:]
        ks.index = [0]
        ks = ks.T
#print(ks)
    else:
        n = r1.shape[0]
        sub = r1.loc[q,:]
        J = sub.rank()
        a_vect = J/len(q)-sub/n
        b_vect = (sub-1)/n-(J-1)/len(q)
        a = a_vect.max()
        b = b_vect.max()
        ks = []
        for i in range(len(a)):
            if a[i] > b[i]:
                ks.append(a[i])
            else:
                ks.append(-b[i])
    return ks
def ranklist(DT):
    # This function takes a panda data frame of gene names and expressions
    # as an input, and output a data frame of gene names and ranks
    ranks = DT.rank(ascending=False, method="first")
    return ranks

    












