import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def get_pairwise_comp_probs(batch_preds, batch_std_labels, sigma=None):
    batch_s_ij = torch.unsqueeze(batch_preds, dim=2) - torch.unsqueeze(batch_preds, dim=1)
    batch_p_ij = torch.sigmoid(sigma * batch_s_ij)

    batch_std_diffs = torch.unsqueeze(batch_std_labels, dim=2) - torch.unsqueeze(batch_std_labels, dim=1)
    batch_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)
    batch_std_p_ij = 0.5 * (1.0 + batch_Sij)

    return batch_p_ij, batch_std_p_ij

def rankloss(score, metric,mask=None,sigma=None):
    gtrank = (-metric).argsort().argsort().float()
    pred,gt = get_pairwise_comp_probs(score, gtrank, sigma=sigma)
    rankloss = F.binary_cross_entropy(input=torch.triu(pred, diagonal=1),
                                     target=torch.triu(gt, diagonal=1), reduction='none')
    rankloss = torch.sum(rankloss, dim=(2, 1))*mask
    return rankloss.mean()

def p_relate(score,mpjpe,eps=1e-12):
    assert score.shape == mpjpe.shape
    score_c = score - score.mean(dim=1).view(-1,1)
    mpjpe_c = mpjpe - mpjpe.mean(dim=1).view(-1,1)
    var_score = torch.sqrt((score_c**2).sum(dim=1))
    var_mpjpe = torch.sqrt((mpjpe_c**2).sum(dim=1))
    conv = (score_c*mpjpe_c).sum(dim=1)
    p_relate = (conv/(var_score*var_mpjpe+eps)).mean()
    loss = (p_relate + 1)
    return loss

def p_relate_mask(score,mpjpe,mask,eps=1e-12):
    assert score.shape == mpjpe.shape
    score_c = score - score.mean(dim=1).view(-1,1)
    mpjpe_c = mpjpe - mpjpe.mean(dim=1).view(-1,1)
    var_score = torch.sqrt((score_c**2).sum(dim=1))
    var_mpjpe = torch.sqrt((mpjpe_c**2).sum(dim=1))
    conv = (score_c*mpjpe_c).sum(dim=1)
    p_relate = (conv/(var_score*var_mpjpe+eps))+1
    loss = p_relate*mask
    return loss.mean()

def flatten_prelate(score,mpjpe,eps=1e-12):
    score = score.view(-1)
    mpjpe = mpjpe.view(-1)
    score_c = score - score.mean()
    mpjpe_c = mpjpe - mpjpe.mean()
    var_score = torch.sqrt((score_c**2).sum())
    var_mpjpe = torch.sqrt((mpjpe_c**2).sum())
    conv = (score_c*mpjpe_c).sum()
    p_relate = conv/(var_score*var_mpjpe+eps)
    loss = (p_relate+1)
    return loss

def spearman(x, y):
    assert x.shape == y.shape
    x = x.argsort().argsort().float()
    y = y.argsort().argsort().float()
    x = (x - x.mean(dim=1).unsqueeze(1)) / x.std(dim=1).unsqueeze(1)
    y = (y - y.mean(dim=1).unsqueeze(1)) / y.std(dim=1).unsqueeze(1)
    return (x * y).mean()

def torch_dcg_at_k(batch_rankings, cutoff=None):
    if cutoff is None: 
        cutoff = batch_rankings.size(1)
    batch_numerators = batch_rankings
    device = batch_rankings.device
    batch_discounts = torch.log2(torch.arange(cutoff, dtype=torch.float, device=device).expand_as(batch_numerators) + 2.0)
    batch_dcg_at_k = torch.sum(batch_numerators/batch_discounts, dim=1, keepdim=True)
    return batch_dcg_at_k

def get_delta_ndcg(batch_ideal_rankings, batch_predict_rankings):
    '''
    Delta-nDCG w.r.t. pairwise swapping of the currently predicted ltr_adhoc
    :param batch_ideal_rankings: the standard labels sorted in a descending order
    :param batch_predicted_rankings: the standard labels sorted based on the corresponding predictions
    :return:
    '''
    # ideal discount cumulative gains
    device = batch_ideal_rankings.device
    batch_idcgs = torch_dcg_at_k(batch_rankings=batch_ideal_rankings)
    batch_gains = batch_predict_rankings
    """
    if LABEL_TYPE.MultiLabel == label_type:
        batch_gains = torch.pow(2.0, batch_predict_rankings) - 1.0
    elif LABEL_TYPE.Permutation == label_type:
        batch_gains = batch_predict_rankings
    else:
        raise NotImplementedError
    """

    batch_n_gains = batch_gains / batch_idcgs               # normalised gains
    batch_ng_diffs = torch.unsqueeze(batch_n_gains, dim=2) - torch.unsqueeze(batch_n_gains, dim=1)

    batch_std_ranks = torch.arange(batch_predict_rankings.size(1), dtype=torch.float, device=device)
    batch_dists = 1.0 / torch.log2(batch_std_ranks + 2.0)   # discount co-efficients
    batch_dists = torch.unsqueeze(batch_dists, dim=0)
    batch_dists_diffs = torch.unsqueeze(batch_dists, dim=2) - torch.unsqueeze(batch_dists, dim=1)
    batch_delta_ndcg = torch.abs(batch_ng_diffs) * torch.abs(batch_dists_diffs)  # absolute changes w.r.t. pairwise swapping

    return batch_delta_ndcg

def lambdarankloss(score, metric,mask=None,sigma=None):
    gtrank = (-metric).argsort().argsort().float()
    pred,gt = get_pairwise_comp_probs(score, gtrank, sigma=sigma)
    rankloss = F.binary_cross_entropy(input=torch.triu(pred, diagonal=1),
                                     target=torch.triu(gt, diagonal=1), reduction='none')
    rankloss = torch.sum(rankloss, dim=(2, 1))*mask

    batch_descending_preds, batch_pred_desc_inds = torch.sort(score, dim=1, descending=True)
    batch_predict_rankings = torch.gather(gtrank, dim=1, index=batch_pred_desc_inds)

    batch_p_ij, batch_std_p_ij = get_pairwise_comp_probs(batch_preds=batch_descending_preds,
                                                        batch_std_labels=batch_predict_rankings,
                                                        sigma=sigma)

    batch_delta_ndcg = get_delta_ndcg(batch_ideal_rankings=gtrank,
                                        batch_predict_rankings=batch_predict_rankings)

    _batch_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1),
                                            target=torch.triu(batch_std_p_ij, diagonal=1),
                                            weight=torch.triu(batch_delta_ndcg, diagonal=1), reduction='none')

    batch_loss = torch.sum(torch.sum(_batch_loss, dim=(2, 1)))
    return batch_loss.mean()