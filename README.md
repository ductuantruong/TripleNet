# TripletNet
This is a pytorch implementation of [Triply Supervised Decoder Networks for Joint Detection and Segmentation](https://arxiv.org/abs/1809.09299) paper. The source code is runnable. However, there are still bugs in our implementations that output weird predictions.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages for preparing the dataset, training and testing the model.

```bash
pip install -r requirements.txt
```

## Dataset
Download dataset [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/), put them under `VOCdevkit` directory as the following directory structure:
```
VOCdevkit
-| VOC2007
   -| Annotations
   -| ImageSets
   -| JPEGImages
   -| SegmentationClass
   -| SegmentationObject
```

## Usage
### Training

```bash
python train.py --model='model name' --run_name='experiment name'
```

Example:
```bash
python train.py --model=pairnet --run_name=pairnet_voc_2007
```
To train TripleNet model, switch `model` to `model=triplenet`

### Testing
```bash
python test.py --model='model name' --model_checkpoint='path to saved model checkpoint'
```

Example:
```bash
python test.py --model=pairnet --model_checkpoint=checkpoints/pairnet_voc_2007/epoch=99-step=36083.ckpt
```

### Pretrained Model
We have uploaded a pretrained model of our experiments. You can download the from OneDrive:
<br/>[Pairnet](https://entuedu-my.sharepoint.com/personal/ductuan001_e_ntu_edu_sg/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fductuan001%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FDL%5FProj%2Fcheckpoints%2Fpairnet%5Fseg%5Fonly2007)
<br/>[TripleNet](https://entuedu-my.sharepoint.com/personal/ductuan001_e_ntu_edu_sg/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fductuan001%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FDL%5FProj%2Fcheckpoints%2Ftriple%5Fseg%5Fonly2007)

Download it and put it into the model_checkpoint folder.

## Reference
Some of the code in `Dataset/` and `utils/` is from the following repos
- [1] [SSD-variants](https://github.com/uoip/SSD-variants)
- [2] [pytorch-segmentation](https://github.com/yassouali/pytorch-segmentation)
