import torch
import torch.nn as nn
import torch.nn.functional as F
# torch.use_deterministic_algorithms(True)

import pytorch_lightning as pl
from Model.model import PairNet
from Model.model import TripleNet

from utils.loss import MultiBoxLoss

class LightningModelPairNet(pl.LightningModule):
    def __init__(self, HPARAMS):
        super().__init__()
        self.save_hyperparameters()

        self.model = PairNet(HPARAMS['n_classes'])
            
        self.det_criterion = MultiBoxLoss()
        self.seg_criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

        self.lr = HPARAMS['lr']
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
        # y_h = torch.stack(y_h).reshape(-1,)
        # y_a = torch.stack(y_a).reshape(-1,)
        # y_g = torch.stack(y_g).reshape(-1,)
        
        loc_hat, det_hat, seg_hat = self(img)

        loc_loss, det_loss = self.det_criterion(loc_hat, det_hat, bboxes, det_labels)
        # seg_loss = self.seg_criterion(seg_hat, seg_labels)
        seg_loss = 0
        for seg_h in seg_hat:
            # try:
            seg_loss_temp = self.seg_criterion(seg_h, seg_labels)
            if not torch.isnan(seg_loss_temp):
                seg_loss += seg_loss_temp
            # except IndexError:
                # print(seg_labels)
                # print(seg_labels.shape, seg_h.shape)
                # seg_labels = torch.as_tensor(seg_labels * 20/254, dtype=torch.long)
                # print(seg_labels)
                # seg_loss_temp = self.seg_criterion(seg_h, seg_labels)
            # if not torch.isnan(seg_loss_temp):
                # seg_loss += seg_loss_temp
        loss = loc_loss + det_loss + seg_loss

        return {
                'loss':loss, 
                'train_loc_loss': det_loss.item(),
                'train_det_loss': det_loss.item(),
                'train_seg_loss': seg_loss,
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

        seg_loss = 0
        for seg_h in seg_hat:
          #try:
            seg_loss_temp = self.seg_criterion(seg_h, seg_labels)
            if not torch.isnan(seg_loss_temp):
                seg_loss += seg_loss_temp
          #except IndexError:
          #  seg_loss += self.seg_criterion(seg_h*20/254,seg_labels)
        # print(loc_hat.shape)
        # print(bboxes.shape)
        loc_loss, det_loss = self.det_criterion(loc_hat, det_hat, bboxes, det_labels)
        val_loss = loc_loss + det_loss + seg_loss

        return {
                'val_loss':val_loss, 
                'val_loc_loss':loc_loss.item(),
                'val_det_loss':det_loss.item(),
                'val_seg_loss':seg_loss
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

    def test_step(self, batch, batch_idx):
        img, bboxes, det_labels, seg_labels = batch
        y_h = torch.stack(y_h).reshape(-1,)
        y_a = torch.stack(y_a).reshape(-1,)
        y_g = torch.stack(y_g).reshape(-1,)
        
        loc_hat, det_hat, seg_hat = self(img)

        loc_loss, det_loss = self.det_criterion(loc_hat, det_hat, bboxes, det_labels)
        seg_loss = self.seg_criterion(seg_hat, seg_labels)
        loss = loc_loss + det_loss + seg_loss

        return {
                'test_loss':loss, 
                'test_loc_loss':loc_loss.item(),
                'test_det_loss':det_loss.item(),
                'test_seg_loss':seg_loss.item()
            }


    def test_epoch_end(self, outputs):
        n_batch = len(outputs)
        test_loss = torch.tensor([x['test_loss'] for x in outputs]).mean()
        loc_loss = torch.tensor([x['test_loc_loss'] for x in outputs]).sum()/n_batch
        det_loss = torch.tensor([x['test_det_loss'] for x in outputs]).sum()/n_batch
        seg_loss = torch.tensor([x['test_seg_loss'] for x in outputs]).sum()/n_batch

        self.log('test/loss', test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/loc', loc_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/det', det_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/seg', seg_loss.item(), on_step=False, on_epoch=True, prog_bar=True)


        pbar = {
                'test/loss':test_loss.item(),
                'test/loc':loc_loss.item(),
                'test/det':det_loss.item(),
                'test/seg':seg_loss.item()
            }
        self.logger.log_hyperparams(pbar)
        self.log_dict(pbar)


class LightningModelTripleNet(pl.LightningModule):
    def __init__(self, HPARAMS):
        super().__init__()
        self.save_hyperparameters()

        self.model = TripleNet(HPARAMS['n_classes'])
            
        self.det_criterion = MultiBoxLoss()
        self.seg_criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

        self.lr = HPARAMS['lr']
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
        img, bboxes, det_labels, seg_labels_class_aware = batch

        loc_hat, det_hat, msf_seg_hat, list_seg_hat_class_aware, list_seg_hat_class_agnost = self(img)

        loc_loss, det_loss = self.det_criterion(loc_hat, det_hat, bboxes, det_labels)

        seg_loss_class_aware = 0.0
        for seg_h in list_seg_hat_class_aware:
            seg_loss_class_aware += self.seg_criterion(seg_h, seg_labels_class_aware)

        seg_labels_class_agnost = self.convert_to_class_agnost(seg_labels_class_aware)
        seg_loss_class_agnost = 0.0
        for seg_h in list_seg_hat_class_agnost:
            seg_loss_class_agnost += self.seg_criterion(seg_h, seg_labels_class_agnost)
        

        loss = loc_loss + det_loss + seg_loss_class_aware + seg_loss_class_agnost


        return {
                'loss':loss, 
                'train_loc_loss': det_loss.item(),
                'train_det_loss': det_loss.item(),
                'train_seg_loss_class_aware': seg_loss_class_aware.item(),
                'train_seg_loss_class_agnost': seg_loss_class_agnost.item(),
            }
    
    def training_epoch_end(self, outputs):
        n_batch = len(outputs)
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
        loc_loss = torch.tensor([x['train_loc_loss'] for x in outputs]).sum()/n_batch
        det_loss = torch.tensor([x['train_det_loss'] for x in outputs]).sum()/n_batch
        seg_loss_class_aware = torch.tensor([x['train_seg_loss_class_aware'] for x in outputs]).sum()/n_batch
        seg_loss_class_agnost = torch.tensor([x['train_seg_loss_class_agnost'] for x in outputs]).sum()/n_batch

        self.log('train/loss' , loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/loc', loc_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/det', det_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/seg_class_aware', seg_loss_class_aware.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/seg_class_agnost', seg_loss_class_agnost.item(), on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img, bboxes, det_labels, seg_labels_class_aware = batch
        
        loc_hat, det_hat, msf_seg_hat, list_seg_hat_class_aware, list_seg_hat_class_agnost = self(img)

        loc_loss, det_loss = self.det_criterion(loc_hat, det_hat, bboxes, det_labels)

        seg_loss_class_aware = 0
        for seg_h in list_seg_hat_class_aware:
            seg_loss_class_aware += self.seg_criterion(seg_h, seg_labels_class_aware)

        seg_labels_class_agnost = self.convert_to_class_agnost(seg_labels_class_aware)
        seg_loss_class_agnost = 0.0
        for seg_h in list_seg_hat_class_agnost:
            seg_loss_class_agnost += self.seg_criterion(seg_h, seg_labels_class_agnost)

        val_loss = loc_loss + det_loss + seg_loss_class_aware + seg_loss_class_agnost

        return {
                'val_loss':val_loss, 
                'val_loc_loss':loc_loss.item(),
                'val_det_loss':det_loss.item(),
                'val_seg_loss_class_aware':seg_loss_class_aware.item(),
                'val_seg_loss_class_agnost':seg_loss_class_agnost.item()
            }

    def validation_epoch_end(self, outputs):
        n_batch = len(outputs)
        val_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        loc_loss = torch.tensor([x['val_loc_loss'] for x in outputs]).sum()/n_batch
        det_loss = torch.tensor([x['val_det_loss'] for x in outputs]).sum()/n_batch
        seg_loss_class_aware = torch.tensor([x['val_seg_loss_class_aware'] for x in outputs]).sum()/n_batch
        seg_loss_class_agnost = torch.tensor([x['val_seg_loss_class_agnost'] for x in outputs]).sum()/n_batch

        self.log('val/loss' , val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/loc', loc_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/det', det_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/seg_class_aware', seg_loss_class_aware.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/seg_class_agnost', seg_loss_class_agnost.item(), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        img, bboxes, det_labels, seg_labels_class_aware = batch
        
        loc_hat, det_hat, msf_seg_hat, list_seg_hat_class_aware, list_seg_hat_class_agnost = self(img)

        loc_loss, det_loss = self.det_criterion(loc_hat, det_hat, bboxes, det_labels)
        seg_loss_class_aware = self.seg_criterion(msf_seg_hat, seg_labels_class_aware)

        seg_labels_class_agnost = self.convert_to_class_agnost(seg_labels_class_aware)
        seg_loss_class_agnost = 0.0
        for seg_h in list_seg_hat_class_agnost:
            seg_loss_class_agnost += self.seg_criterion(seg_h, seg_labels_class_agnost)

        loss = loc_loss + det_loss + seg_loss_class_aware + seg_loss_class_agnost

        return {
                'test_loss':loss, 
                'test_loc_loss':loc_loss.item(),
                'test_det_loss':det_loss.item(),
                'test_seg_loss_class_aware':seg_loss_class_aware.item(),
                'test_seg_loss_class_agnost':seg_loss_class_agnost.item()
            }


    def test_epoch_end(self, outputs):
        n_batch = len(outputs)
        test_loss = torch.tensor([x['test_loss'] for x in outputs]).mean()
        loc_loss = torch.tensor([x['test_loc_loss'] for x in outputs]).sum()/n_batch
        det_loss = torch.tensor([x['test_det_loss'] for x in outputs]).sum()/n_batch
        seg_loss_class_aware = torch.tensor([x['test_seg_loss_class_aware'] for x in outputs]).sum()/n_batch
        seg_loss_class_agnost = torch.tensor([x['test_seg_loss_class_agnost'] for x in outputs]).sum()/n_batch

        self.log('test/loss', test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/loc', loc_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/det', det_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/seg_class_aware', seg_loss_class_aware.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/seg_class_agnost', seg_loss_class_agnost.item(), on_step=False, on_epoch=True, prog_bar=True)


        pbar = {
                'test/loss':test_loss.item(),
                'test/loc':loc_loss.item(),
                'test/det':det_loss.item(),
                'test/seg_class_aware':seg_loss_class_aware.item(),
                'test/seg_class_agnost':seg_loss_class_agnost.item()
            }
        self.logger.log_hyperparams(pbar)
        self.log_dict(pbar)

    # Converts given original ground truth labels to class agnostic version
    def convert_to_class_agnost(self, seg_labels):

        # torch.set_printoptions(profile="full")
        # print(seg_labels)

        # Change Bg to '0' Class and non-Bg to '1' class
        seg_labels_class_agnost = seg_labels.clone()
        seg_labels_class_agnost[seg_labels==255] = 0
        seg_labels_class_agnost[seg_labels!=255] = 1

        #print(seg_labels_class_agnost)

        return seg_labels_class_agnost
