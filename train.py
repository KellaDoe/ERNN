# %%
import argparse
import time
import os

from sklearn.metrics import (precision_score,
                             recall_score,
                             f1_score,
                             classification_report)
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm
from evaluation import *
from datasets import ISIC
from models import EMENN
import warnings

warnings.filterwarnings('ignore')

def check_dir(test_dir):
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)


def valid_id(outputs_, labels_):
    pred = torch.argmax(outputs_, dim=1).detach().cpu().numpy()
    target = torch.argmax(labels_, dim=1).detach().cpu().numpy()
    result = {'pre': precision_score(target, pred, average='weighted'),
              'rec': recall_score(target, pred, average='weighted'),
              'f1s': f1_score(target, pred, average='weighted')}
    return result


def exp_rampup(epoch, warmup):
    if warmup == 0:
        return 1.0
    else:
        current = np.clip(epoch, 0, warmup)
        phase = 1.0 - current / warmup
        return float(np.exp(-5.0 * phase * phase))


def zero_cosine_rampdown(current, epochs):
    return float(.5 * (1 + np.cos(current * np.pi / epochs)))


if __name__ == '__main__':
    manual_seed = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('-cuda', type=str, default='2', help='choose cuda, default 2')
    parser.add_argument('-save', action='store_true', help='whether model is saved, default false')
    parser.add_argument('-focal', type=float, default=0)
    parser.add_argument('-kl', type=float, default=0)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    p_train_img = '/mnt/mnt_data/Data/ISIC_2019/ISIC_2019_Training_Input'
    p_train_label = '/mnt/mnt_data/Data/ISIC_2019/ISIC_2019_Training_GroundTruth.csv'
    class_split = ([0, 1, 2, 3, 4, 7], [5, 6])
    num_class = len(class_split[0])
    batch_size = 256
    lr = 1e-4
    wd = 1e-4
    max_epochs = 100
    logs_dir = './runs/isic_{}_ERNN'.format(time.strftime('%Y-%m-%d_%H_%M', time.localtime()))

    isic = ISIC(img_dir=p_train_img, label_dir=p_train_label)
    for fold in range(5):
        log_dir = os.path.join(logs_dir, f'fold_{fold}')
        txt_dir = os.path.join(log_dir, 'log.txt')
        tensor_dir = os.path.join(log_dir, 'tensor')
        model_dir = os.path.join(log_dir, 'models')
        check_dir(tensor_dir)
        check_dir(model_dir)

        # experiment initialization
        torch.manual_seed(manual_seed)
        with open(txt_dir, 'a') as f:
            f.write(f'Parameters:\n')
            f.write(f'-focal:{args.focal}\n')
            f.write(f'-kl:{args.kl}\n')

        train_set, valid_set_id, valid_set_ood = isic.split(fold, class_split=class_split)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader_id = DataLoader(valid_set_id, batch_size=batch_size)
        valid_loader_ood = DataLoader(valid_set_ood, batch_size=batch_size)

        # MENN is ERNN
        model = EMENN(in_dim=3, out_dim=num_class, focal=args.focal, alpha_kl=args.kl).to(device)

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
        summary_writer = SummaryWriter(log_dir)
        metrics = ['pre', 'rec', 'f1s']
        valid_id_best = {}
        for metric in metrics:
            valid_id_best[metric] = {'value': 0,
                                     'epoch': 0}
        valid_pent = {'value': 0,
                     'epoch': 0}

        for epoch in range(max_epochs + 1):
            # eval_epoch()
            model.eval()
            with torch.no_grad():
                with tqdm(total=len(valid_loader_id), ncols=70) as _tqdm:
                    _tqdm.set_description(f'Validating: e{epoch + 1}')
                    features_i, outputs_i, probs_i, labels_i = [], [], [], []
                    valid_loss = 0
                    for data, label in valid_loader_id:
                        data = Variable(data).to(device).float()
                        label = Variable(label).to(device)

                        feature, output, prob = model(data)
                        loss = model.criterion(feature, output, label)
                        valid_loss += loss

                        features_i.append(feature)
                        outputs_i.append(output)
                        probs_i.append(prob)
                        labels_i.append(label)
                        _tqdm.update(1)

            valid_loss /= len(valid_loader_id)
            summary_writer.add_scalars('Loss', {'valid_loss': valid_loss}, epoch)

            features_ = torch.cat(features_i, dim=0)
            outputs_i = torch.cat(outputs_i, dim=0)
            probs_i = torch.cat(probs_i, dim=0)
            labels_i = torch.cat(labels_i, dim=0)

            result = valid_id(probs_i, labels_i)
            print('current metric: ', end='')
            for key in result.keys():
                current = result[key]
                if valid_id_best.__contains__(key):
                    if current >= valid_id_best[key]['value']:
                        valid_id_best[key]['value'] = current
                        valid_id_best[key]['epoch'] = epoch
                        if args.save:
                            torch.save(model, f'{model_dir}/best_{key}_distance.pth')
                            print(f'best {key} model saved in epoch {epoch}!')
                    best_value = valid_id_best[key]['value']
                    best_epoch = valid_id_best[key]['epoch']
                    print(f'{key}: {current:.4f}({best_value:.4f} in {best_epoch})', end='')
            print()

            # eval_ood
            model.eval()
            with torch.no_grad():
                with tqdm(total=len(valid_loader_ood), ncols=70) as _tqdm:
                    _tqdm.set_description(f'Validating: ood in {epoch + 1}')
                    features_o, outputs_o, probs_o, labels_o = [], [], [], []
                    valid_loss = 0
                    for data, label in valid_loader_ood:
                        data = Variable(data).to(device).float()
                        label = Variable(label).to(device)
                        feature, output, prob = model(data)
                        features_o.append(feature)
                        outputs_o.append(output)
                        probs_o.append(prob)
                        labels_o.append(label)
                        _tqdm.update(1)
                features_o = torch.cat(features_o, dim=0)
                outputs_o = torch.cat(outputs_o, dim=0)
                probs_o = torch.cat(probs_o, dim=0)
                labels_o = torch.cat(labels_o, dim=0)

            labels_i_np = labels_i.detach().cpu().numpy()
            probs_i_np = probs_i.detach().cpu().numpy()
            print(f'valid in epoch {epoch}:')
            print(classification_report(np.argmax(labels_i_np, axis=1), np.argmax(probs_i_np, axis=1)))

            print('pent_ood_metric')
            pent_i = probs_i.detach().cpu().numpy()
            pent_o = probs_o.detach().cpu().numpy()
            pent_i = np.sum(np.log(pent_i) * pent_i, axis=1)
            pent_o = np.sum(np.log(pent_o) * pent_o, axis=1)
            result_pent = metric_ood(pent_i, pent_o)['Bas']
            summary_writer.add_histogram('pent_id', pent_i, epoch)
            summary_writer.add_histogram('pent_ood', pent_o, epoch)
            summary_writer.add_histogram('pent', np.concatenate((pent_o, pent_i)), epoch)

            if result_pent['AUROC'] >= valid_pent['value']:
                valid_pent['value'] = result_pent['AUROC']
                valid_pent['epoch'] = epoch
                if args.save:
                    torch.save(model, f'{model_dir}/best_ent_auroc.pth')
                    print(f'best ent model saved in epoch {epoch}!')
            summary_writer.add_scalars('AUROC', {'PENT': result_pent['AUROC'], }, epoch)

            with open(txt_dir, 'a') as f:
                f.write(f'valid in epoch {epoch}:\n')
                f.write(classification_report(np.argmax(labels_i_np, axis=1), np.argmax(probs_i_np, axis=1)))
                f.write('\n')

                f.write('-msp_ood_metric\n')
                f.write(str(result_pent))
                f.write('\n')
                f.write('\n\n')

            if epoch == max_epochs:
                continue
            # train_epoch()
            model.train()
            with tqdm(total=len(train_loader), ncols=70) as _tqdm:
                _tqdm.set_description(f'Training: e{epoch + 1}')
                train_loss = 0
                for data, label in train_loader:
                    data = Variable(data).to(device).float()
                    label = Variable(label).to(device)

                    feature, output, prob = model(data)
                    loss = model.criterion(feature, output, label)

                    train_loss += loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    _tqdm.set_postfix(loss='{:.6f}'.format(loss.item()))
                    _tqdm.update(1)
                train_loss /= len(train_loader)
                summary_writer.add_scalars('Loss', {'train_loss': train_loss}, epoch)

            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4 * zero_cosine_rampdown(epoch, max_epochs)

        with open(txt_dir, 'a') as f:
            f.write('\n\n*************Fold Result*************')
            for key in valid_id_best:
                f.write('save best {} model in epoch {} with value {}\n'.format(key,
                                                                                valid_id_best[key]['epoch'],
                                                                                valid_id_best[key]['value']))
            f.write('save best ent ood model at epoch {} with value {}\n'.format(valid_pent['epoch'],
                                                                                 valid_pent['value']))
