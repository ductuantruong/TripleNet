# TripleNet
This is a pytorch implementation of the TripleNet and PairNet models from the paper [Triply Supervised Decoder Networks for Joint Detection and Segmentation](https://arxiv.org/abs/1809.09299).

**_NOTE:_**  If you want to run the models with semantic segmentation task only, you can checkout the `seg_only_models` branch and follow the README in that branch.

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
We have uploaded pretrained models of our experiments. You can download them from [OneDrive]:
<br/>[Pairnet](https://entuedu-my.sharepoint.com/:u:/g/personal/ductuan001_e_ntu_edu_sg/EbnskdqK0FpDkif15KxTtmIBu5y6c7gZ1NZYb_XJD6t3NQ?e=Jbh6bT)
<br/>[TripleNet](https://entuedu-my.sharepoint.com/:u:/g/personal/ductuan001_e_ntu_edu_sg/Ef01-t_YdxZIqkXq6bQ9aKMBPtI-8vDAWgJTrnldTDfacA?e=r7pmw9)

Download it and put it into the model_checkpoint folder.

## Reference
Most of the code in `Dataset/` and `utils/` is from the following repos
- [1] [SSD-variants](https://github.com/uoip/SSD-variants)
- [2] [pytorch-segmentation](https://github.com/yassouali/pytorch-segmentation)
