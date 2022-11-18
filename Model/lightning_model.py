import torch
import torch.nn as nn
import torch.nn.functional as F
# torch.use_deterministic_algorithms(True)

import pytorch_lightning as pl
from Model.model import PairNet
from Model.model import TripleNet

from utils.loss import MultiBoxLoss
from enum import Enum

# Model Names Enum
# For use with --model param in train/test script 
class ModelNames(str, Enum):
    PairNet = 'pairnet'
    TripleNet = 'triplenet'

class LightningModelPairNet(pl.LightningModule):
    def __init__(self, HPARAMS):
        super().__init__()
        self.save_hyperparameters()

        self.model = PairNet(HPARAMS['n_classes'], HPARAMS['aspect_ratios'])
            
        self.det_criterion = MultiBoxLoss()
        self.seg_criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

        self.lr = HPARAMS['lr']
        self.training_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Model Details: #Params = {self.count_total_parameters()}\t#Trainable Params = {self.count_trainable_parameters()}")

    def count_total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        optimizer.step(closure=optimizer_closure)
        if self.trainer.global_step > 60:
            lr_scale = 0.1
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

    def training_step(self, batch, batch_idx):
        img, bboxes, det_labels, seg_labels = batch
        
        loc_hat, det_hat, seg_hat = self(img)

        loc_loss, det_loss = self.det_criterion(loc_hat, det_hat, bboxes, det_labels)
        seg_loss = torch.tensor([0], dtype=torch.float).to(self.training_device)
        for seg_h in seg_hat:
            seg_loss_temp = self.seg_criterion(seg_h, seg_labels)
            if not torch.isnan(seg_loss_temp):
                seg_loss += seg_loss_temp

        loss = loc_loss + det_loss + seg_loss

        return {
                'loss':loss, 
                'train_loc_loss': loc_loss.item(),
                'train_det_loss': det_loss.item(),
                'train_seg_loss': seg_loss.item(),
            }
    
    def training_epoch_end(self, outputs):
        n_batch = len(outputs)
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
        loc_loss = torch.tensor([x['train_loc_loss'] for x in outputs]).sum()/n_batch
        det_loss = torch.tensor([x['train_det_loss'] for x in outputs]).sum()/n_batch
        seg_loss = torch.tensor([x['train_seg_loss'] for x in outputs]).sum()/n_batch

        self.log('train/loss' , loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/loc', loc_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/det', det_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/seg', seg_loss.item(), on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img, bboxes, det_labels, seg_labels = batch
        
        loc_hat, det_hat, seg_hat = self(img)

        seg_loss = torch.tensor([0], dtype=torch.float).to(self.training_device)
        for seg_h in seg_hat:
            seg_loss_temp = self.seg_criterion(seg_h, seg_labels)
            if not torch.isnan(seg_loss_temp):
                seg_loss += seg_loss_temp

        loc_loss, det_loss = self.det_criterion(loc_hat, det_hat, bboxes, det_labels)
        val_loss = loc_loss + det_loss + seg_loss

        return {
                'val_loss':val_loss, 
                'val_loc_loss':loc_loss.item(),
                'val_det_loss':det_loss.item(),
                'val_seg_loss':seg_loss.item()
            }

    def validation_epoch_end(self, outputs):
        n_batch = len(outputs)
        val_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        loc_loss = torch.tensor([x['val_loc_loss'] for x in outputs]).sum()/n_batch
        det_loss = torch.tensor([x['val_det_loss'] for x in outputs]).sum()/n_batch
        seg_loss = torch.tensor([x['val_seg_loss'] for x in outputs]).sum()/n_batch

        self.log('val/loss' , val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/loc', loc_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/det', det_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/seg', seg_loss.item(), on_step=False, on_epoch=True, prog_bar=True)

class LightningModelTripleNet(pl.LightningModule):
    def __init__(self, HPARAMS):
        super().__init__()
        self.save_hyperparameters()

        self.model = TripleNet(HPARAMS['n_classes'], HPARAMS['aspect_ratios'])
            
        self.det_criterion = MultiBoxLoss()
        self.seg_criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

        self.lr = HPARAMS['lr']
        self.n_classes = HPARAMS['n_classes']
        self.training_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Model Details: #Params = {self.count_total_parameters()}\t#Trainable Params = {self.count_trainable_parameters()}")

    def count_total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return [optimizer]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        optimizer.step(closure=optimizer_closure)
        if self.trainer.global_step > 60:
            lr_scale = 0.1
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

    def training_step(self, batch, batch_idx):
        img, bboxes, det_labels, seg_labels = batch

        loc_hat, det_hat, list_seg_hat, seg_hat_msf, list_seg_hat_clsag = self(img)

        # Loc and Det Loss
        loc_loss, det_loss = self.det_criterion(loc_hat, det_hat, bboxes, det_labels)

        # Loss for Multi-scaled Fusion
        seg_loss_msf = torch.tensor([0], dtype=torch.float).to(self.training_device)
        seg_loss_tmp = self.seg_criterion(seg_hat_msf, seg_labels)
        if not torch.isnan(seg_loss_tmp):
            seg_loss_msf = seg_loss_tmp

        # Loss for standard segmentation (per layer/class aware)
        seg_loss = torch.tensor([0], dtype=torch.float).to(self.training_device)
        for seg_h in list_seg_hat:
            seg_loss_tmp = self.seg_criterion(seg_h, seg_labels)
            if not torch.isnan(seg_loss_tmp):
                seg_loss += seg_loss_tmp

        # Loss for class agnostic segmentation
        seg_labels_clsag = self.convert_to_class_agnost(seg_labels)
        seg_loss_clsag = torch.tensor([0], dtype=torch.float).to(self.training_device)
        for seg_h in list_seg_hat_clsag:
            seg_loss_tmp = self.seg_criterion(seg_h, seg_labels_clsag)
            if not torch.isnan(seg_loss_tmp):
                seg_loss_clsag += seg_loss_tmp
        
        loss = loc_loss + det_loss + seg_loss_msf + seg_loss + seg_loss_clsag

        return {
                'loss':loss, 
                'train_loc_loss': loc_loss.item(),
                'train_det_loss': det_loss.item(),
                'train_seg_loss': seg_loss.item(),
                'train_seg_loss_msf': seg_loss_msf.item(),
                'train_seg_loss_clsag': seg_loss_clsag.item(),
            }
    
    def training_epoch_end(self, outputs):
        n_batch = len(outputs)
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
        loc_loss = torch.tensor([x['train_loc_loss'] for x in outputs]).sum()/n_batch
        det_loss = torch.tensor([x['train_det_loss'] for x in outputs]).sum()/n_batch
        seg_loss = torch.tensor([x['train_seg_loss'] for x in outputs]).sum()/n_batch
        seg_loss_msf = torch.tensor([x['train_seg_loss_msf'] for x in outputs]).sum()/n_batch
        seg_loss_clsag = torch.tensor([x['train_seg_loss_clsag'] for x in outputs]).sum()/n_batch

        self.log('train/loss' , loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/loc', loc_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/det', det_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/seg', seg_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/seg_msf', seg_loss_msf.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/seg_clsag', seg_loss_clsag.item(), on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img, bboxes, det_labels, seg_labels = batch
        
        loc_hat, det_hat, list_seg_hat, seg_hat_msf, list_seg_hat_clsag = self(img)

        # Loc and Det Loss
        loc_loss, det_loss = self.det_criterion(loc_hat, det_hat, bboxes, det_labels)

        # Loss for Multi-scaled Fusion
        seg_loss_msf = torch.tensor([0]).to(self.training_device)
        seg_loss_tmp = self.seg_criterion(seg_hat_msf, seg_labels)
        if not torch.isnan(seg_loss_tmp):
            seg_loss_msf = seg_loss_tmp

        # Loss for standard segmentation (per layer/class aware)
        seg_loss = torch.tensor([0], dtype=torch.float).to(self.training_device)
        for seg_h in list_seg_hat:
            seg_loss_tmp = self.seg_criterion(seg_h, seg_labels)
            if not torch.isnan(seg_loss_tmp):
                seg_loss += seg_loss_tmp

        # Loss for class agnostic segmentation
        seg_labels_clsag = self.convert_to_class_agnost(seg_labels)
        seg_loss_clsag = torch.tensor([0], dtype=torch.float).to(self.training_device)
        for seg_h in list_seg_hat_clsag:
            seg_loss_tmp = self.seg_criterion(seg_h, seg_labels_clsag)
            if not torch.isnan(seg_loss_tmp):
                seg_loss_clsag += seg_loss_tmp

        val_loss = loc_loss + det_loss + seg_loss_msf + seg_loss + seg_loss_clsag

        return {
                'val_loss':val_loss, 
                'val_loc_loss':loc_loss.item(),
                'val_det_loss':det_loss.item(),
                'val_seg_loss':seg_loss.item(),
                'val_seg_loss_msf':seg_loss_msf.item(),
                'val_seg_loss_clsag':seg_loss_clsag.item()
            }

    def validation_epoch_end(self, outputs):
        n_batch = len(outputs)
        val_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        loc_loss = torch.tensor([x['val_loc_loss'] for x in outputs]).sum()/n_batch
        det_loss = torch.tensor([x['val_det_loss'] for x in outputs]).sum()/n_batch
        seg_loss = torch.tensor([x['val_seg_loss'] for x in outputs]).sum()/n_batch
        seg_loss_msf = torch.tensor([x['val_seg_loss_msf'] for x in outputs]).sum()/n_batch
        seg_loss_clsag = torch.tensor([x['val_seg_loss_clsag'] for x in outputs]).sum()/n_batch

        self.log('val/loss' , val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/loc', loc_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/det', det_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/seg', seg_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/seg_msf', seg_loss_msf.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/seg_clsag', seg_loss_clsag.item(), on_step=False, on_epoch=True, prog_bar=True)

    # Converts given original ground truth labels to class agnostic version
    def convert_to_class_agnost(self, seg_labels):

        # Bg Class remain at '0'
        # All other classes set to '1'
        seg_labels_class_agnost = seg_labels.clone()
        for i in range(1,self.n_classes + 1):
            seg_labels_class_agnost[seg_labels_class_agnost == i] = 1

        return seg_labels_class_agnost
