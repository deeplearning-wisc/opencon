import argparse
import warnings
import torch
import torch.optim as optim
import open_world_cifar as datasets
from utils import cluster_acc, AverageMeter, entropy, MarginLoss, accuracy, TransformTwice
from sklearn import metrics
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle

def train(args, algo, device, train_label_loader, train_unlabel_loader, optimizer, m, epoch, tf_writer):
    algo.model.train()
    algo.m = -min(m, 0.5)

    unlabel_loader_iter = cycle(train_unlabel_loader)

    for batch_idx, ((x, x2), target) in enumerate(train_label_loader):
        ((ux, ux2), target_unlabeled) = next(unlabel_loader_iter)
        x = torch.cat([x, ux], 0)
        x2 = torch.cat([x2, ux2], 0)
        # labeled_len = len(target)

        x, x2, target, target_unlabeled = x.to(device), x2.to(device), target.to(device), target_unlabeled.to(device)
        optimizer.zero_grad()
        loss = algo.forward_cifar(x, x2, target, target_unlabeled=target_unlabeled)
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
                epoch, batch_idx + 1, len(train_label_loader), prob_msg,
                losses_simclr = algo.simclr_losses,
                losses_supcon = algo.supcon_losses,
                losses_semicon = algo.semicon_losses,
                losses_ent=algo.entropy_losses,
            ))


    tf_writer.add_scalar('loss/entropy', algo.entropy_losses.avg, epoch)
    tf_writer.add_scalar('loss/simclr', algo.simclr_losses.avg, epoch)
    tf_writer.add_scalar('loss/supcon', algo.supcon_losses.avg, epoch)
    tf_writer.add_scalar('loss/semicon', algo.semicon_losses.avg, epoch)


def test(args, algo, labeled_num, device, test_loader, epoch, tf_writer):
    algo.model.eval()
    preds = np.array([])
    targets = np.array([])
    confs = np.array([])
    features = []
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            ret_dict = algo.forward_cifar(x, None, label, evalmode=True)
            pred = ret_dict['label_pseudo']
            conf = ret_dict['conf']
            feat = ret_dict['features']
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
            features.append(feat.data.cpu().numpy())

    targets = targets.astype(int)
    preds = preds.astype(int)

    seen_mask = targets < args.labeled_num
    unseen_mask = ~seen_mask

    if (epoch + 1) % args.save_freq == 0:
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


def main():
    parser = argparse.ArgumentParser(description='orca')
    parser.add_argument('--milestones', nargs='+', type=int, default=[100, 150])
    parser.add_argument('--dataset', default='cifar100', help='dataset setting')
    parser.add_argument('--proto-num', default=100, type=int)
    parser.add_argument('--labeled-num', default=50, type=int)
    parser.add_argument('--labeled-ratio', default=0.5, type=float)
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--exp_root', type=str, default='./results/')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size')

    parser.add_argument('--momentum-proto', type=float, default=0.9)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--vis-freq', type=int, default=1e5)
    parser.add_argument('--save-freq', type=int, default=100)
    parser.add_argument('--id_thresh', type=int, default=70)
    parser.add_argument('--name', type=str, default='opencon')
    parser.add_argument('--w-semicon', type=float, default=0.1)
    parser.add_argument('--w-supcon', type=float, default=0.2)
    parser.add_argument('--w-simclr', type=float, default=1.)
    parser.add_argument('--w-ent', default=0.05, type=float)
    parser.add_argument('--temp_simclr', default=.4, type=float)
    parser.add_argument('--temp_supcon', default=0.1, type=float)
    parser.add_argument('--temp_semicon', default=0.7, type=float)
    parser.add_argument('--lr', default=2e-2, type=float)
    parser.add_argument('--mask_policy', default="zuzuzoodzood", type=str)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    args.name += f"-{args.dataset}"
    args.savedir = os.path.join(args.exp_root, args.name, )
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
        os.makedirs(os.path.join(args.savedir, 'snapshot'))
        os.makedirs(os.path.join(args.savedir, 'vis'))

    args.savedir = args.savedir + '/'
    args.device = device

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'cifar10':
        train_label_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']))
        train_unlabel_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']), unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=datasets.dict_transform['cifar_test'], unlabeled_idxs=train_label_set.unlabeled_idxs)
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        train_label_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']))
        train_unlabel_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']), unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=datasets.dict_transform['cifar_test'], unlabeled_idxs=train_label_set.unlabeled_idxs)
        args.num_classes = 100
    else:
        warnings.warn('Dataset is not listed')
        return

    labeled_len = len(train_label_set)
    unlabeled_len = len(train_unlabel_set)
    labeled_batch_size = int(args.batch_size * labeled_len / (labeled_len + unlabeled_len))

    # Initialize the splits
    train_label_loader = torch.utils.data.DataLoader(train_label_set, batch_size=labeled_batch_size, shuffle=True, num_workers=2, drop_last=True)
    train_unlabel_loader = torch.utils.data.DataLoader(train_unlabel_set, batch_size=args.batch_size - labeled_batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=1)


    from models.OpenSupCon import OpenSupCon
    algo = OpenSupCon("RN18_simclr_CIFAR", args)

    # Set the optimizer
    optimizer = optim.SGD(algo.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    tf_writer = SummaryWriter(log_dir=args.savedir)


    mean_uncert = 0
    max_w_semicon = args.w_semicon
    for epoch in range(args.epochs):
        algo.reset_stat()

        args.w_semicon = min(max_w_semicon, max_w_semicon / 100 * epoch)
        print(args.w_semicon)

        train(args, algo, device, train_label_loader, train_unlabel_loader, optimizer, mean_uncert, epoch, tf_writer)
        mean_uncert = test(args, algo, args.labeled_num, device, test_loader, epoch, tf_writer)
        scheduler.step()


if __name__ == '__main__':
    main()
