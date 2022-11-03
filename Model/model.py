import torch
import torch.nn as nn
import torch.nn.functional as F 

from torchvision.models import resnet50
from torchvision.models.resnet import Bottleneck
from collections import OrderedDict


class BottleneckBlock(nn.Module):
    def __init__(self, inchannels):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, inchannels//4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inchannels//4)
        self.conv2 = nn.Conv2d(inchannels//4, inchannels//4, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inchannels//4)
        self.conv3 = nn.Conv2d(inchannels//4, inchannels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(inchannels)
        self.relu = nn.ReLU(inplace=True)
        self.shorcut = nn.Sequential(
                nn.Conv2d(inchannels, inchannels, kernel_size=1, bias=False),
                nn.BatchNorm2d(inchannels))
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.shorcut(x)
        out += identity
        out = self.relu(out)
        return out

class DecoderLayer(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, stride=1):
        super().__init__()

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels1 + in_channels2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels))
        
        
        self.residual_layer = nn.Sequential(
            nn.Conv2d(in_channels1 + in_channels2, out_channels // 4, 1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x, encoder_x):
        x = F.upsample(x, size=encoder_x.size()[2:], mode='bilinear')
        x = torch.cat([x, encoder_x], dim=1)
        shortcut = self.shortcut(x)
        residual = self.residual_layer(x)
        return F.relu(shortcut + residual)



class PairNet(nn.Module):
    def __init__(self, n_classes, image_size=300, x4=True):
        super().__init__()
        self.n_classes = n_classes
        self.image_size = image_size
        if image_size == 300:
            self.config = self.config300(x4)
        elif image_size == 512:
            self.config = self.config512(x4)

        self.skip_layers = self.config['skip_layers']
        self.pred_layers = self.config['pred_layers']

        self.Base = resnet50(pretrained=True)


        self.resnet = resnet50(pretrained=True)
        self.conv1 = nn.Sequential(*list(self.resnet.children())[:4])
        self.res1_4 = nn.Sequential(OrderedDict([
            ('res1', list(self.resnet.children())[4]),
            ('res2', list(self.resnet.children())[5]),
            ('res3', list(self.resnet.children())[6]),
            ('res4', list(self.resnet.children())[7])]
        ))
        self.res5_7 = nn.Sequential(OrderedDict([
            ('res5', BottleneckBlock(2048)),
            ('res6', BottleneckBlock(2048)),
            ('res7', BottleneckBlock(2048))])
        )

        self._initialize_weights(self.res5_7)

        self.encoder = nn.Sequential(self.res1_4, self.res5_7)

        self.last_encoder_conv = nn.Conv2d(2048, 512, 1, bias=False)

        self.decoder = nn.Sequential(OrderedDict([
            ('decoder1', DecoderLayer(2048, 2048, 512)),
            ('decoder2', DecoderLayer(512, 2048, 512)),
            ('decoder3', DecoderLayer(512,  2048, 512)),
            ('decoder4', DecoderLayer(512,  2048, 512)),
            ('decoder5', DecoderLayer(512,  1024, 512))]
        ))
            # ('decoder_layer2', DecoderLayer(512,  512,  512)),
            # ('decoder_layer1', DecoderLayer(512,  256,  512))]))

        n_boxes = len(self.config['aspect_ratios']) + 1
        self.list_localized_head = nn.ModuleList([])
        self.list_detector_head = nn.ModuleList([])
        for i in range(len(self.config['grids'])):
            self.list_localized_head.append(nn.Conv2d(512, n_boxes * 4, 3, padding=1))
            self.list_detector_head.append(nn.Conv2d(512, n_boxes * (self.n_classes + 1), 3, padding=1))

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(512, self.n_classes, kernel_size=3, stride=1, padding=1)
        )


    def _initialize_weights(self, block):
        for module in block.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
    
    def forward(self, x):
        list_encoder_embedding = list()
        
        x = self.conv1(x)

        for layer_name, encoder_layer in self.res1_4._modules.items():
            x = encoder_layer(x)
            if layer_name in self.skip_layers:
                list_encoder_embedding.append(x)

        for layer_name, encoder_layer in self.res5_7._modules.items():
            x = encoder_layer(x)
            if layer_name in self.skip_layers:
                list_encoder_embedding.append(x)

        list_encoder_embedding = list_encoder_embedding[::-1]

        list_decoder_embedding = [self.last_encoder_conv(x)]

        for i, (name, m) in enumerate(self.decoder._modules.items()):
            x = m(x, list_encoder_embedding[i])
            list_decoder_embedding.append(x)

        loc_hat, det_hat = self.detection_prediction(list_decoder_embedding) 
        return loc_hat, det_hat, self.segmentation_prediction(list_decoder_embedding)


    def detection_prediction(self, xs):
        locs = []
        confs = []
        for i, x in enumerate(xs):
            loc = self.list_localized_head[i](x) # if isinstance(self.list_localized_head, nn.ModuleList) else self.Loc(x)
            loc = loc.permute(0, 2, 3, 1).contiguous().view(loc.size(0), -1, 4)
            locs.append(loc)

            conf = self.list_detector_head[i](x) if isinstance(self.list_detector_head, nn.ModuleList) else self.list_detector_head(x)
            conf = conf.permute(0, 2, 3, 1).contiguous().view(conf.size(0), -1, self.n_classes + 1)
            confs.append(conf)
        return torch.cat(locs, dim=1), torch.cat(confs, dim=1)

    def segmentation_prediction(self, xs):
        list_seg_hat = []
        for x in xs:
            out = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=True)
            out = self.segmentation_head(out)
            list_seg_hat.append(out)
        return list_seg_hat

    def config300(self, x4=False):
        config = {
            'skip_layers': ['res2', 'res3', 'res4', 'res5', 'res6'],
            'pred_layers': ['decoder1', 'decoder2', 'decoder3', 'decoder4', 'decoder5'],
            'name': 'PairNet300-resnet50-Det' + '-s4' if x4 else '-s8',
            'image_size': 300,
            'grids': [75]*x4 + [38, 19, 10, 5, 3, 1],
            'sizes': [s / 300. for s in [30, 60, 111, 162, 213, 264, 315]],
            'aspect_ratios': (1/4., 1/3.,  1/2.,  1,  2,  3),
            'batch_size': 32,
            'init_lr': 1e-4,
            'stepvalues': (35000, 50000),    
            'max_iter': 65000
        }
        return config

    def config512(self, x4=False):
        config = {
            'skip_layers': ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'layer7', 'layer8'],
            'pred_layers': ['rev_layer7', 'rev_layer6', 'rev_layer5', 'rev_layer4', 'rev_layer3', 'rev_layer2'] + (
                            ['rev_layer1']*x4),
            'name': 'PairNet512-resnet50-Det' + '-s4' if x4 else '-s8',
            'image_size': 512,
            'grids': [128]*x4 + [64, 32, 16, 8, 4, 2, 1],
            'sizes': [s / 512. for s in [20.48, 61.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72]],
            'aspect_ratios': (1/3.,  1/2.,  1,  2,  3),
            'batch_size': 16,
            'init_lr': 1e-4,
            'stepvalues': (45000, 60000),
            'max_iter': 75000
        }
        return config




