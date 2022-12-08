import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F
import models
from models.AbstractAlgo import Algo
from utils import AverageMeter


class OpenSupCon(Algo):

    def __init__(self, model_type, args):
        super(OpenSupCon, self).__init__(model_type, args)
        self.proj_layer = nn.Sequential(
                nn.Linear(self.featdim, self.featdim),
                nn.ReLU(inplace=True),
                nn.Linear(self.featdim, 128)
            ).to(args.device)

        self.register_buffer("proto", self.normalizer(torch.randn((self.args.proto_num, 128), dtype=torch.float)).to(args.device))
        self.contrast_mode = 'all'

    @torch.no_grad()
    def sync_prototype(self):
        self.proto.data = self.proto_tmp.data

    def cl_loss(self, cosine_dist, mask_pos, mask_neg, mask_ind, temperature):
        cosine_mat = torch.div(cosine_dist, temperature)
        mat_max, _ = torch.max(cosine_mat, dim=1, keepdim=True)
        cosine_mat = cosine_mat - mat_max.detach()
        pos_term = (cosine_mat * mask_pos).sum(1) / (mask_pos.sum(1) + 1e-10)
        neg_term = (torch.exp(cosine_mat) * mask_neg).sum(1)
        log_term = (pos_term - torch.log(neg_term + 1e-15)) * mask_ind
        return -log_term.sum()

    def loss_stat_init(self, ):
        super(OpenSupCon, self).loss_stat_init()
        self.simclr_loss = AverageMeter('simclr_loss', ':.4e')
        self.supcon_loss = AverageMeter('supcon_loss', ':.4e')
        self.semicon_loss = AverageMeter('semicon_loss', ':.4e')

    def indexing_mask_factory(self, cosine_dist, target, label_concat, logit, target_unlabeled=None):
        bsz = len(cosine_dist) // 2
        labeled_len = len(target)

        # Batch Size 512
        # Concated Batch Size 1024
        # Labeled Indexing 0-128 & 512-640
        # Labeled Indexing 128-512 & 640-1024

        ######## GET INDEX MASK IN CONCAT BATCH ##########
        unlabeled_ind_mask = torch.zeros(bsz * 2).to(cosine_dist.device)
        unlabeled_ind_mask[labeled_len:bsz] = 1.
        unlabeled_ind_mask[bsz+labeled_len:] = 1.
        labeled_ind_mask = 1 - unlabeled_ind_mask

        ############ OOD assignment by Groudtruth
        ood_pred_mask_oracle = (torch.cat([target, target_unlabeled]) >= self.args.labeled_num).repeat(2)

        ############ OOD assignment by fake label
        ood_pred_mask_naive = unlabeled_ind_mask * (label_concat >= self.args.labeled_num).repeat(2)

        ############ OOD assignment by ID-ness score
        id_score = logit[:, :self.args.labeled_num].max(1)[0]
        ood_pred_mask = unlabeled_ind_mask * (id_score < np.percentile(id_score[:labeled_len].data.cpu().numpy(), 100 - int(self.args.id_thresh))).repeat(2)
        id_pred_mask = unlabeled_ind_mask * (id_score > np.percentile(id_score[:labeled_len].data.cpu().numpy(), 100 - 10)).repeat(2)

        dict_ret = {
            "zuzuzoodzood": {
                "simclr_pos_idxmask": unlabeled_ind_mask,
                "simclr_neg_idxmask": unlabeled_ind_mask,
                "semicon_pos_idxmask": ood_pred_mask,
                "semicon_neg_idxmask": ood_pred_mask,
                "supcon_pos_idxmask": labeled_ind_mask,
                "supcon_neg_idxmask": labeled_ind_mask,
            },
        }[self.args.mask_policy]

        dict_ret["unlabeled_ind_mask"] = unlabeled_ind_mask
        return dict_ret



    def opensupcon_loss(self, feat_proj, feat_proj2, target, label_concat, logit, target_unlabeled=None):
        bsz = len(feat_proj)
        feat_proj_cat = torch.cat([feat_proj, feat_proj2])
        labeled_len = len(target)

        cosine_dist = feat_proj_cat @ feat_proj_cat.t()

        idxnmask = self.indexing_mask_factory(cosine_dist, target, label_concat, logit, target_unlabeled)
        simclr_pos_idxmask = idxnmask['simclr_pos_idxmask']
        simclr_neg_idxmask = idxnmask['simclr_neg_idxmask']
        supcon_pos_idxmask = idxnmask['supcon_pos_idxmask']
        supcon_neg_idxmask =  idxnmask['supcon_neg_idxmask']
        semicon_pos_idxmask = idxnmask['semicon_pos_idxmask']
        semicon_neg_idxmask = idxnmask['semicon_neg_idxmask']


        mask_neg_base = 1 - torch.diag(torch.ones(2 * bsz)).to(self.args.device)
        ##############################  SimCLR    ##############################
        mask_pos_sm =  torch.diag(torch.ones(bsz)).to(self.args.device)
        mask_pos_simclr = mask_pos_sm.repeat(2, 2) * mask_neg_base
        mask_neg_simclr = mask_neg_base * (simclr_neg_idxmask[:, None] * simclr_neg_idxmask[None, :])      ####### zu only

        simclr_loss = self.cl_loss(cosine_dist, mask_pos_simclr, mask_neg_simclr, simclr_pos_idxmask, self.args.temp_simclr)
        # simclr_loss_ref = self.simclr_loss_ref(feat_proj_cat)

        ##############################  SupCon    ##############################
        mask_pos_sm =  torch.zeros(bsz, bsz).to(self.args.device)
        target_ = target.contiguous().view(-1, 1)
        mask_pos_sm[:labeled_len, :labeled_len] = torch.eq(target_, target_.T)
        mask_pos_supcon = mask_pos_sm.repeat(2, 2) * mask_neg_base
        mask_neg_supcon = mask_neg_base * (supcon_neg_idxmask[:, None] * supcon_neg_idxmask[None, :])           ####### Zl only

        supcon_loss = self.cl_loss(cosine_dist, mask_pos_supcon, mask_neg_supcon, supcon_pos_idxmask, self.args.temp_supcon)

        ##############################  SemiSupCon        ##############################
        # mask_pos_sm = ((label_concat > self.args.labeled_num)[:, None] * (label_concat > self.args.labeled_num)[None, :] * torch.eq(label_concat[:, None], label_concat[None, :])).to(self.args.device)
        mask_pos_sm = torch.eq(label_concat[:, None], label_concat[None, :]).to(self.args.device)
        mask_pos_semicon = mask_pos_sm.repeat(2, 2) * mask_neg_base * (semicon_pos_idxmask[:, None] * semicon_pos_idxmask[None, :])
        mask_neg_semicon = mask_neg_base * (semicon_neg_idxmask[:, None] * semicon_neg_idxmask[None, :])

        semicon_loss = self.cl_loss(cosine_dist, mask_pos_semicon, mask_neg_semicon, semicon_pos_idxmask, self.args.temp_semicon)

        cl_loss = simclr_loss * self.args.w_simclr / ((simclr_pos_idxmask.sum() + 1e-10)) + \
                  supcon_loss * self.args.w_supcon / (supcon_pos_idxmask.sum())  +  \
                  semicon_loss * self.args.w_semicon / (semicon_pos_idxmask.sum() + 1e-10)

        self.simclr_losses.update(simclr_loss.item() / (simclr_pos_idxmask.sum() + 1e-10).item(), self.args.batch_size)
        self.supcon_losses.update(supcon_loss.item() / supcon_pos_idxmask.sum().item(), self.args.batch_size)
        self.semicon_losses.update(semicon_loss.item() / (semicon_pos_idxmask.sum() + 1e-10).item(), self.args.batch_size)

        return cl_loss



    def entropy(self, x, q=1):
        """
        Helper function to compute the entropy over the batch
        input: batch w/ shape [b, num_classes]
        output: entropy value [is ideally -log(num_classes)]
        """
        EPS = 1e-5
        x_ = torch.clamp(x, min=EPS)
        b = x_ * torch.log(x_ / q)

        if len(b.size()) == 2:  # Sample-wise entropy
            return - b.sum(dim=1).mean()
        elif len(b.size()) == 1:  # Distribution-wise entropy
            return - b.sum()
        else:
            raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))

    def forward(self, x, x2, target, evalmode=False, target_unlabeled=None):
        return self.forward_cifar(x, x2, target, evalmode, target_unlabeled)

    def forward_cifar(self, x, x2, target, evalmode=False, target_unlabeled=None):
        if evalmode:
            _, feat = self.model(x)
            feat_proj = self.normalizer(self.proj_layer(feat))
            logit = feat_proj @ self.proto.data.T
            prob = F.softmax(logit, dim=1)
            conf, pred = prob.max(1)
            return {
                "logit": logit,
                "features_penul": self.normalizer(feat),
                "features": feat_proj,
                "conf": conf,
                "label_pseudo": pred,
            }

        _, feat = self.model(x)
        feat_proj = self.normalizer(self.proj_layer(feat))
        _, feat2 = self.model(x2)
        feat_proj2 = self.normalizer(self.proj_layer(feat2))
        labeled_len = len(target)

        dist = feat_proj @ self.proto.data.T * 10
        dist2 = feat_proj2 @ self.proto.data.T * 10


        if labeled_len == 0:
            label_pseudo = dist[:, self.args.labeled_num:].argmax(1) + self.args.labeled_num
        else:
            ################################# BALANCED DISTANCE ######################################
            balanced_dist = (dist.clone() - dist.mean(1, keepdims=True)) / dist.std(1, keepdims=True)
            label_pseudo = balanced_dist.argmax(1).type(torch.int)
        label_concat = torch.cat([target, label_pseudo[labeled_len:]]).type(torch.int)

        # Unlabeling filtering
        cl_loss = self.opensupcon_loss(feat_proj, feat_proj2, target, label_concat, balanced_dist, target_unlabeled=target_unlabeled)

        if labeled_len == 0:
            ent_loss = self.entropy(torch.softmax(dist[:, self.args.labeled_num:], dim=1).mean(0))
        else:
            q = torch.Tensor([1 - self.args.labeled_ratio] * self.args.labeled_num + [1] * (self.args.proto_num - self.args.labeled_num)).to(self.args.device)
            q = q / q.sum()
            ent_loss = self.entropy(torch.softmax(dist[labeled_len:], dim=1).mean(0), q)

        loss = cl_loss - self.args.w_ent * ent_loss

        self.update_label_stat(label_pseudo[labeled_len:])

        if not evalmode:
            updated_ind = torch.cat([torch.ones_like(target), (label_pseudo[labeled_len:] >= self.args.labeled_num)]).type(torch.bool)
            rand_order = np.random.choice(updated_ind.sum().item(), updated_ind.sum().item(), replace=False)
            self.update_prototype_lazy(feat_proj[updated_ind][rand_order], label_concat[updated_ind][rand_order], momemt=self.args.momentum_proto)

        self.entropy_losses.update(ent_loss.item(), self.args.batch_size)
        return loss


