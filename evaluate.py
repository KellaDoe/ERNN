from sklearn.metrics import (precision_score,
                             recall_score,
                             f1_score,
                             classification_report)
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

from evaluation import *
from datasets import ISIC
from models import EMENN


def valid_id(outputs_, labels_):
    pred = torch.argmax(outputs_, dim=1).detach().cpu().numpy()
    target = torch.argmax(labels_, dim=1).detach().cpu().numpy()
    result = {'pre': precision_score(target, pred, average='weighted'),
              'rec': recall_score(target, pred, average='weighted'),
              'f1s': f1_score(target, pred, average='weighted')}
    return result


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    p_train_img = '/mnt/mnt_data/ISIC_2019/ISIC_2019_Training_Input'
    p_train_label = '/mnt/mnt_data/ISIC_2019/ISIC_2019_Training_GroundTruth.csv'
    p_model = 'best_metric.pth'
    model = torch.load(p_model)
    class_split = ([0, 1, 2, 3, 4, 7], [5, 6])
    batch_size = 256
    isic = ISIC(img_dir=p_train_img, label_dir=p_train_label)

    for fold in range(5):
        train_set, valid_set_id, valid_set_ood = isic.split(fold, class_split=class_split)
        valid_loader_id = DataLoader(valid_set_id, batch_size=batch_size)
        valid_loader_ood = DataLoader(valid_set_ood, batch_size=batch_size)
        model.eval()
        with torch.no_grad():
            with tqdm(total=len(valid_loader_id), ncols=70) as _tqdm:
                _tqdm.set_description(f'Validating: id: ')
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

            features_ = torch.cat(features_i, dim=0)
            outputs_i = torch.cat(outputs_i, dim=0)
            probs_i = torch.cat(probs_i, dim=0)
            labels_i = torch.cat(labels_i, dim=0)
            result = valid_id(probs_i, labels_i)

            model.eval()
            with torch.no_grad():
                with tqdm(total=len(valid_loader_ood), ncols=70) as _tqdm:
                    _tqdm.set_description(f'Validating: ood: ')
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

            print(f'valid id fold {fold}:')
            print(classification_report(np.argmax(labels_i_np, axis=1), np.argmax(probs_i_np, axis=1)))

            print(f'valid ood in fold {fold}:')
            pent_i = probs_i.detach().cpu().numpy()
            pent_o = probs_o.detach().cpu().numpy()
            pent_i = np.sum(np.log(pent_i) * pent_i, axis=1)
            pent_o = np.sum(np.log(pent_o) * pent_o, axis=1)
            result_pent = metric_ood(pent_i, pent_o)['Bas']
