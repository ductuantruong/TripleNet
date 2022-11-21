from argparse import ArgumentParser
from multiprocessing import Pool
import os
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torch
import torch.utils.data as data
import random
from numpy.random import RandomState
import numpy as np

from Model.lightning_model import LightningModelPairNet, LightningModelTripleNet, ModelNames

from Dataset.dataset import VOC, VOCDataset

from utils.multibox import MultiBox
from utils.transform import *
from utils.metric import *

from tqdm import tqdm 
from mean_average_precision import MetricBuilder
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

# Convert batch data into cuda type if is_gpu flag is set
def preprocess_batch(batch, is_gpu=False):
    img, bboxes, det_labels, seg_labels = batch
    if is_gpu:
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        return img.to(dev), bboxes.to(dev), det_labels.to(dev), seg_labels.to(dev)
    
    return img, bboxes, det_labels, seg_labels


if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--voc_root', type=str, default='VOCdevkit')
    parser.add_argument('--voc_year', type=str, default='2007')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=480)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--dev', type=str, default=False)
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--n_classes', type=int, default=20)
    parser.add_argument('--model', type=str, default=ModelNames.PairNet.value)
    parser.add_argument('--model_checkpoint', type=str, default=None)
    parser.add_argument('--upstream_model', type=str, default=None)
    parser.add_argument('--sizes', type=list, default=[s / 300. for s in [30, 60, 111, 162, 213, 264, 315]])
    parser.add_argument('--aspect_ratios', type=list, default=(1/4., 1/3.,  1/2.,  1,  2,  3))
    parser.add_argument('--grids', type=list, default=[38, 19, 10, 5, 3, 2])
    parser.add_argument('--run_name', type=str, default='first_try')

    parser = pl.Trainer.add_argparse_args(parser)
    cfg = parser.parse_args()
    cfg = vars(cfg)

    print('Training Model on TIMIT Dataset\n#Cores = {}\t#GPU = {}'.format(cfg['n_workers'], cfg['gpu']))

    encoder = MultiBox(cfg)

    transform = Compose([
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
        image_set=[(cfg['voc_year'], 'test')],
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

    ## Model
    if cfg['model'] == ModelNames.PairNet.value:
        model = LightningModelPairNet.load_from_checkpoint(cfg['model_checkpoint'], HPARAMS=cfg, strict=False)
        print("Model: PairNet")
    elif cfg['model'] == ModelNames.TripleNet.value:
        model = LightningModelTripleNet.load_from_checkpoint(cfg['model_checkpoint'], HPARAMS=cfg)
        print("Model: TripleNet")
    else:
        print("ERROR: Invalid model in parameters.")
        sys.exit()

    ## CPU or GPU
    is_gpu = False
    if cfg['gpu'] != 0:
        is_gpu = True
        

    if is_gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    model.to(device)
    model.eval()

    # Segmentation Evaluation Vars
    total_pix_correct = 0   # Total no. of Correct Pixels
    total_pix_labelled = 0  # Total no. of Labelled Pixels
    arr_intersec = np.full(cfg['n_classes'], 0.)    # Array of intersection area with GT per class 
    arr_union = np.full(cfg['n_classes'], 0.)       # Array of union area with GT per class 
    i = 0

    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=cfg['n_classes'])
    for batch in tqdm(testloader):
        img, bboxes, det_labels, seg_labels = preprocess_batch(batch, is_gpu)

        # np_bboxes = bboxes.squeeze(0).cpu().numpy()
        # np_bboxes = np.multiply(np_bboxes, 100)

        # np_det_labels = det_labels.squeeze(0).cpu().numpy()
        # gt = np.concatenate((np_bboxes, np.expand_dims(np_det_labels, 1), np.zeros((np_bboxes.shape[0], 2))), axis=1)

        # loc_hat, det_hat, seg_hat = model.model(img, is_eval=True)
        seg_hat = model.model(img, is_eval=True)

        b_pix_correct, b_n_label, b_intersec, b_uninion = seg_eval_metrics(seg_hat, seg_labels, cfg['n_classes'])
        total_pix_correct += b_pix_correct
        total_pix_labelled += b_n_label
        arr_intersec = np.add(arr_intersec, b_intersec)
        arr_union =  np.add(arr_union, b_uninion)


        # loc_hat = loc_hat.data.cpu().numpy()[0]
        # det_hat = det_hat.data.cpu().numpy()[0]

        # boxes, labels, scores = encoder.decode(loc_hat, det_hat, nms_thresh=0.5, conf_thresh=0.01)
        # boxes = np.multiply(boxes, 100)
        # pred = np.concatenate((boxes, np.expand_dims(labels, 1), np.expand_dims(scores, 1)), axis=1)

        # metric_fn.add(pred, gt)

    # print(f"VOC PASCAL mAP in all points: {metric_fn.value(iou_thresholds=0.5)['mAP']}")
    print(eval_voc_segmentation(arr_intersec, arr_union, total_pix_correct, total_pix_labelled))
        