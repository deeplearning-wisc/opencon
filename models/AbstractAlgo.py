import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F
import models

from utils import AverageMeter

class Algo(nn.Module):

    def __init__(self, model_type, args):
        super(Algo, self).__init__()
        self.args = args
        self.model = self.load_model(model_type)
        self.featdim = self.model.featdim
        self.bce = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.momem_proto = args.momentum_proto

        self.status = {"dense_sample_count": AverageMeter()}

        self.normalizer = lambda x: x / torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-10

        # self.register_buffer("proto", self.normalizer(torch.randn((self.args.proto_num, 128), dtype=torch.float)).to(args.device))
        self.register_buffer("label_stat", torch.zeros(self.args.proto_num, dtype=torch.int))

        self.loss_stat_init()

    def loss_stat_init(self, ):
        self.bce_losses = AverageMeter('bce_loss', ':.4e')
        self.ce_losses = AverageMeter('ce_loss', ':.4e')
        self.entropy_losses = AverageMeter('entropy_loss', ':.4e')
        self.cls_losses = AverageMeter('cls_losses', ':.4e')
        self.cl_losses = AverageMeter('cl_losses', ':.4e')
        self.feat_losses = AverageMeter('feat_losses', ':.4e')
        self.disp_losses = AverageMeter('disp_losses', ':.4e')
        self.simclr_losses = AverageMeter('simclr_loss', ':.4e')
        self.supcon_losses = AverageMeter('supcon_loss', ':.4e')
        self.semicon_losses = AverageMeter('semicon_loss', ':.4e')


    def load_model(self, type="RN50_simclr"):
        model = None
        if type == "RN50_supcon":
            model = models.resnet50(num_classes=self.args.num_classes)
            ckpt = torch.load('./pretrained/supcon_imagenet1k.pth')
            state_dict = {k.replace('module.encoder.', ''): v for k, v in
                 ckpt['model'].items()}
            model.load_state_dict(state_dict, strict=False)

            for name, param in model.named_parameters():
                if 'fc' not in name and 'layer4' not in name:
                    param.requires_grad = False

            model = model.to(self.args.device)
            model.featdim = 2048


        if type == "RN50_simclr_1k":
            model = models.resnet50(num_classes=self.args.num_classes)
            ckpt = torch.load('./pretrained/simclr_imagenet1k_phase799.torch')
            state_dict = {k.replace('_feature_blocks.', ''): v for k, v in
                 ckpt['classy_state_dict']['base_model']['model']['trunk'].items()}
            model.load_state_dict(state_dict, strict=False)

            for name, param in model.named_parameters():
                if 'fc' not in name and 'layer4' not in name:
                    param.requires_grad = False

            model = model.to(self.args.device)
            model.featdim = 2048

        if type == "RN50_simclr":
            model = models.resnet50(num_classes=self.args.num_classes)
            state_dict = torch.load('./pretrained/simclr_imagenet_100.pth.tar')
            # state_dict = torch.load('/home/sunyiyou/workspace/KNN/checkpoints/supcon.pth')['model']
            # state_dict = {k.replace('module.encoder.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)

            for name, param in model.named_parameters():
                if 'fc' not in name and 'layer4' not in name:
                    param.requires_grad = False

            model = model.to(self.args.device)
            model.featdim = 2048

        if type == "RN18_simclr_CIFAR":
            model = models.resnet18(num_classes=self.args.num_classes)
            model = model.to(self.args.device)
            if self.args.dataset == 'cifar10':
                state_dict = torch.load('./pretrained/simclr_cifar_10.pth.tar')
            elif self.args.dataset == 'cifar100':
                state_dict = torch.load('./pretrained/simclr_cifar_100.pth.tar')
            model.load_state_dict(state_dict, strict=False)
            model.featdim = 512

            for name, param in model.named_parameters():
                if 'linear' not in name and 'layer4' not in name:
                    param.requires_grad = False
            model = model.to(self.args.device)

        if type == "VIT":
            from models import vision_transformer as vits

            model = vits.__dict__['vit_base']()

            state_dict = torch.load('pretrained/dino_vitbase16_pretrain.pth', map_location='cpu')
            model.load_state_dict(state_dict)

            model.projection_head = vits.__dict__['DINOHead'](in_dim=768,
                                                        out_dim=128, nlayers=3)
            model.featdim = 768

            # HOW MUCH OF BASE MODEL TO FINETUNE
            # ----------------------
            for m in model.parameters():
                m.requires_grad = False

            # Only finetune layers from block 'args.grad_from_block' onwards
            for name, m in model.named_parameters():
                if 'block' in name:
                    block_num = int(name.split('.')[1])
                    if block_num >= 11:
                        m.requires_grad = True
            model = model.to(self.args.device)
        return model

    @torch.no_grad()
    def update_label_stat(self, label):
        self.label_stat += label.bincount(minlength=self.args.proto_num).to(self.label_stat.device)

    @torch.no_grad()
    def reset_stat(self):
        self.label_stat = torch.zeros(self.args.proto_num, dtype=torch.int).to(self.label_stat.device)
        self.loss_stat_init()

    @torch.no_grad()
    def sync_prototype(self):
        pass

    def update_prototype_lazy(self, feat, label, weight=None, momemt=None):
        if momemt is None:
            momemt = self.momem_proto
        self.proto_tmp = self.proto.clone().detach()
        if weight is None:
            weight = torch.ones_like(label)
        for i, l in enumerate(label):
            alpha = 1 - (1. - momemt) * weight[i]
            self.proto_tmp[l] = self.normalizer(alpha * self.proto_tmp[l].data + (1. - alpha) * feat[i])

    def entropy(self, x):
        """
        Helper function to compute the entropy over the batch
        input: batch w/ shape [b, num_classes]
        output: entropy value [is ideally -log(num_classes)]
        """
        EPS = 1e-5
        x_ = torch.clamp(x, min=EPS)
        b = x_ * torch.log(x_)

        if len(b.size()) == 2:  # Sample-wise entropy
            return - b.sum(dim=1).mean()
        elif len(b.size()) == 1:  # Distribution-wise entropy
            return - b.sum()
        else:
            raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))

    def dispersion_loss(self):
        proto = self.proto_tmp
        logits = torch.div(torch.matmul(proto, proto.T),1.)

        rng = torch.arange(0, proto.shape[0]).cuda()
        rng = rng.contiguous().view(-1, 1)
        mask = (1 - torch.eq(rng, rng.T).float()).to(proto.device)

        mean_prob_neg = torch.log((mask * torch.exp(logits)).sum(1) / mask.sum(1))
        loss = mean_prob_neg.max()
        return loss

    def compact_loss(self, dist):
        labels = dist.argmax(1)
        logits_dist = torch.div(dist, self.temperature)
        logits_max, _ = torch.max(logits_dist, dim=1, keepdim=True)
        logits_dist = logits_dist - logits_max.detach()
        logits_mask = torch.zeros_like(logits_dist)
        logits_mask[np.arange(len(labels)), labels] = 1.
        exp_logits = torch.exp(logits_dist)
        log_prob = (logits_dist * logits_mask).sum(1) - torch.log(exp_logits.sum(1))
        loss = -log_prob.mean()
        return loss

    def simclr_loss(self, cosine_mat):
        cosine_mat = torch.div(cosine_mat, self.temperature)
        mat_max, _ = torch.max(cosine_mat, dim=1, keepdim=True)
        cosine_mat = cosine_mat - mat_max.detach()
        diag_mask = torch.diag(torch.ones(len(cosine_mat))).to(cosine_mat.device)
        log_term = (cosine_mat * diag_mask).sum(1) - torch.log(torch.exp(cosine_mat).sum(1))
        loss = -log_term.mean()
        return loss

    def simclr_loss2(self, mat1, mat2):
        mat1 = torch.div(mat1, self.temperature)
        mat_max, _ = torch.max(mat1, dim=1, keepdim=True)
        mat1 = mat1 - mat_max.detach()

        mat2 = torch.div(mat2, self.temperature)
        mat_max, _ = torch.max(mat2, dim=1, keepdim=True)
        mat2 = mat2 - mat_max.detach()

        diag_mask = torch.diag(torch.ones(len(mat1))).to(mat1.device)
        log_term = (mat1 * diag_mask).sum(1) - torch.log((torch.exp(mat1)).sum(1) + (torch.exp(mat2) * (1 - diag_mask)).sum(1))
        loss = -log_term.mean()
        return loss

    def forward_unsup_debug(self, x, x2, label_novel, evalmode=False):
        pass

    def forward(self, x, x2, target, evalmode=False):
        pass
