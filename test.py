from argparse import ArgumentParser
from multiprocessing import Pool
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torch
import torch.utils.data as data
import random
from numpy.random import RandomState
import numpy as np

from Model.lightning_model import LightningModel

from Dataset.dataset import VOC, VOCDataset

from utils.multibox import MultiBox
from utils.transform import *
from utils.metric import *

from tqdm import tqdm 

# SEED
def seed_torch(seed=100):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    pl.utilities.seed.seed_everything(seed)

seed_torch()


if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--voc_root', type=str, default='VOCdevkit')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=480)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dev', type=str, default=False)
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--n_classes', type=int, default=20)
    # parser.add_argument('--model_checkpoint', type=str, default=None)
    parser.add_argument('--upstream_model', type=str, default=None)
    parser.add_argument('--x4', type=bool, default=False)
    parser.add_argument('--sizes', type=list, default=[s / 300. for s in [30, 60, 111, 162, 213, 264, 315]])
    parser.add_argument('--aspect_ratios', type=list, default=(1/4., 1/3.,  1/2.,  1,  2,  3))
    parser.add_argument('--run_name', type=str, default='first_try')
    parser.add_argument('--model_checkpoint', type=str, default='checkpoints/')

    parser = pl.Trainer.add_argparse_args(parser)
    cfg = parser.parse_args()
    cfg = vars(cfg)
    # cfg['grids'] = [75]*cfg['x4'] + [38, 19, 10, 5, 3, 1]
    cfg['grids'] =[38,19,10,5,3,2]

    print('Training Model on TIMIT Dataset\n#Cores = {}\t#GPU = {}'.format(cfg['n_workers'], cfg['gpu']))

    encoder = MultiBox(cfg)

    transform = Compose([
            [ColorJitter(prob=0.5)],  # or write [ColorJitter(), None]
            BoxesToCoords(),
            Resize(300),
            CoordsToBoxes(),
            [SubtractMean(mean=VOC.MEAN)],
            [RGB2BGR()],
            [ToTensor()],
            ])

    ## Test Dataset
    test_set = VOCDataset(
        root=cfg['voc_root'], 
        image_set=[('2007', 'val')],
        keep_difficult=True,
        transform=transform,
        target_transform=None
    )

    ## Validation Dataloader
    testloader = data.DataLoader(
        test_set, 
        batch_size=1,
        shuffle=False, 
        num_workers=cfg['n_workers']
    )

    # model = LightningModel.load_from_checkpoint(cfg['model_checkpoint'], HPARAMS=cfg)
    model = LightningModel(cfg)
    model.to('cuda')
    model.eval()

    gt_bboxes = []
    gt_labels = []
    pix_acc = []
    m_IoU = []
    pred_bboxes = []
    pred_labels = []
    pred_scores = []

    for batch in tqdm(testloader):
        img, bboxes, det_labels, seg_labels = batch
        img, bboxes, det_labels, seg_labels = img.cuda(), bboxes.cuda(), det_labels.cuda(), seg_labels.cuda()
        gt_bboxes.append(bboxes)
        gt_labels.append(det_labels)

        loc_hat, det_hat, seg_hat = model(img, is_eval=True)
        b_pix_ccc, b_IoU, _,_ = seg_eval_metrics(seg_hat, seg_labels, cfg['n_classes'])
        pix_acc.append(b_pix_ccc)
        pix_acc.append(b_IoU)

        loc_hat = loc_hat.data.cpu().numpy()[0]
        det_hat = det_hat.data.cpu().numpy()[0]

        boxes, labels, scores = encoder.decode(loc_hat, det_hat, nms_thresh=0.5, conf_thresh=0.01)

        pred_bboxes.append(boxes)
        pred_labels.append(labels)
        pred_scores.append(scores)

    print(eval_voc_detection(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, iou_thresh=0.5, use_07_metric=True))

        