import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import open_world_imagenet as datasets
import torchvision.transforms as transforms
from utils import cluster_acc, accuracy, TransformTwice
from sklearn import metrics
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter


def train(args, algo, device, train_loader, optimizer, m, labeled_len, epoch, tf_writer):
    algo.model.train()
    algo.m = -min(m, 0.5)

    for batch_idx, ((x, x2), combined_target, idx) in enumerate(train_loader):
        
        target = combined_target[:labeled_len]
        target_unlabeled = combined_target[labeled_len:].to(device)
        x, x2, target = x.to(device), x2.to(device), target.to(device)
        
        loss = algo.forward(x, x2, target, target_unlabeled=target_unlabeled)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        algo.sync_prototype()


        if (batch_idx + 1) % args.print_freq == 0:
            prob_msg = "\t".join([f"{val * 100:.0f}" for val in
                                  list((algo.label_stat / (1e-6 + algo.label_stat.sum())).data.cpu().numpy())])
            print('Train: [{0}][{1}/{2}]\t'
                  'losses_simclr {losses_simclr.val:.3f} ({losses_simclr.avg:.3f})\t'
                  'losses_supcon {losses_supcon.val:.3f} ({losses_supcon.avg:.3f})\t'
                  'losses_semicon {losses_semicon.val:.3f} ({losses_semicon.avg:.3f})\t'
                  'loss_ent {losses_ent.val:.3f} ({losses_ent.avg:.3f})\t'
                  'prob {3}\t'.format(
                epoch, batch_idx + 1, len(train_loader), prob_msg,
                losses_simclr = algo.simclr_losses,
                losses_supcon = algo.supcon_losses,
                losses_semicon = algo.semicon_losses,
                cls_losses=algo.cls_losses,
                losses_ent=algo.entropy_losses,
            ))

    tf_writer.add_scalar('loss/entropy', algo.entropy_losses.avg, epoch)
    tf_writer.add_scalar('loss/simclr', algo.simclr_losses.avg, epoch)
    tf_writer.add_scalar('loss/supcon', algo.supcon_losses.avg, epoch)
    tf_writer.add_scalar('loss/semicon', algo.semicon_losses.avg, epoch)


def test(args, algo, device, test_loader, epoch, tf_writer):
    algo.model.eval()

    preds = np.array([])
    targets = np.array([])
    confs = np.array([])
    with torch.no_grad():
        for batch_idx, (x, label, _) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            optimizer.zero_grad()

            ret_dict = algo.forward_cifar(x, None, label, evalmode=True)
            pred = ret_dict['label_pseudo']
            conf = ret_dict['conf']
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())

    targets = targets.astype(int)
    preds = preds.astype(int)
    seen_mask = targets < args.labeled_num
    unseen_mask = ~seen_mask

    if (epoch+1) % args.save_freq == 0:
        torch.save({
            'epoch': epoch,
            'state_dict': algo.state_dict(),
        }, f"{args.savedir}/snapshot/{epoch}.pth")


    overall_acc = cluster_acc(preds, targets)
    seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
    unseen_nmi = metrics.normalized_mutual_info_score(targets[unseen_mask], preds[unseen_mask])
    mean_uncert = 1 - np.mean(confs)
    print('Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(overall_acc, seen_acc, unseen_acc))
    tf_writer.add_scalar('acc/overall', overall_acc, epoch)
    tf_writer.add_scalar('acc/seen', seen_acc, epoch)
    tf_writer.add_scalar('acc/unseen', unseen_acc, epoch)
    tf_writer.add_scalar('nmi/unseen', unseen_nmi, epoch)
    tf_writer.add_scalar('uncert/test', mean_uncert, epoch)
    return mean_uncert

parser = argparse.ArgumentParser(
            description='orca',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='imagenet100', help='dataset setting')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                        help='device ids assignment (e.g 0 1 2 3)')
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--proto-num', default=100, type=int)

parser.add_argument('--momentum-proto', type=float, default=0.9)
parser.add_argument('--print-freq', type=int, default=10)
parser.add_argument('--vis-freq', type=int, default=1000000)
parser.add_argument('--save-freq', type=int, default=25)
parser.add_argument('--id_thresh', type=int, default=70)
parser.add_argument('--w-semicon', type=float, default=0.1)
parser.add_argument('--w-supcon', type=float, default=0.2)
parser.add_argument('--w-simclr', type=float, default=1)
parser.add_argument('--w-ent', default=0.05, type=float)
parser.add_argument('--temp_simclr', default=0.6, type=float)
parser.add_argument('--temp_supcon', default=0.1, type=float)
parser.add_argument('--temp_semicon', default=0.7, type=float)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--mask_policy', default="zuzuzoodzood", type=str)

parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('--milestones', nargs='+', type=int, default=[60, 90])
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--dataset_root', default='/home/sunyiyou/dataset/imagenet/train', type=str)
parser.add_argument('--exp_root', type=str, default='./results/')
parser.add_argument('--labeled-num', default=50, type=int)
parser.add_argument('--labeled-ratio', default=0.5, type=float)
parser.add_argument('--model_name', type=str, default='resnet')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--num_classes', default=100, type=int)
parser.add_argument('--name', type=str, default='opencon')
args = parser.parse_args()


def clip_model_loader():
    class clip_mix(nn.Module):

        def __init__(self, body):
            super(clip_mix, self).__init__()
            self.body = body
            self.linear = NormedLinear(1024, args.num_classes)

        def forward(self, x):
            feat = self.body(x)
            out = self.linear(feat)
            return out, feat

    import clip
    from models.resnet import NormedLinear
    model_clip, preprocess = clip.load("RN50", device=device, jit=False)
    body = model_clip.visual
    body.float()
    body.dtype = torch.float32
    model = clip_mix(body)
    return model

if __name__ == "__main__":

    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.name += f"-{args.dataset}"
    args.savedir = os.path.join(args.exp_root, args.name, )
    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(os.path.join(args.savedir, 'snapshot'), exist_ok=True)
    os.makedirs(os.path.join(args.savedir, 'vis'), exist_ok=True)

    args.device = device

    from models.OpenSupCon import OpenSupCon
    algo = OpenSupCon("RN50_simclr", args)

    transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    train_label_set = datasets.ImageNetDataset(root=args.dataset_root, anno_file='./data/ImageNet100_label_{}_{:.2f}.txt'.format(args.labeled_num, args.labeled_ratio), transform=TransformTwice(transform_train))
    train_unlabel_set = datasets.ImageNetDataset(root=args.dataset_root, anno_file='./data/ImageNet100_unlabel_{}_{:.2f}.txt'.format(args.labeled_num, args.labeled_ratio), transform=TransformTwice(transform_train))
    concat_set = datasets.ConcatDataset((train_label_set, train_unlabel_set))
    labeled_idxs = range(len(train_label_set))
    unlabeled_idxs = range(len(train_label_set), len(train_label_set)+len(train_unlabel_set))
    batch_sampler = datasets.TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, int(args.batch_size * len(train_unlabel_set) / (len(train_label_set) + len(train_unlabel_set))))

    test_unlabel_set = datasets.ImageNetDataset(root=args.dataset_root, anno_file='./data/ImageNet100_unlabel_{}_{:.2f}.txt'.format(args.labeled_num, args.labeled_ratio), transform=transform_test)

    train_loader = torch.utils.data.DataLoader(concat_set, batch_sampler=batch_sampler, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_unlabel_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    # Set the optimizer
    optimizer = optim.SGD(algo.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    tf_writer = SummaryWriter(log_dir=args.savedir)

    mean_uncert = 0
    max_w_semicon = args.w_semicon

    for epoch in range(args.epochs):
        algo.reset_stat()

        args.w_semicon = min(max_w_semicon, max_w_semicon / 100 * epoch)
        print(args.w_semicon)

        train(args, algo, device, train_loader, optimizer, mean_uncert, batch_sampler.primary_batch_size, epoch, tf_writer)
        mean_uncert = test(args, algo, device, test_loader, epoch, tf_writer)
        scheduler.step()
