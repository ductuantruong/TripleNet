import torch
import torch.nn as nn
import torch.nn.functional as F 

from torchvision.models import resnet50
from torchvision.ops import SqueezeExcitation
from collections import OrderedDict


class BottleneckBlock(nn.Module):
    def __init__(self, inchannels):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, inchannels//4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inchannels//4)
        self.conv2 = nn.Conv2d(inchannels//4, inchannels//4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inchannels//4)
        self.conv3 = nn.Conv2d(inchannels//4, inchannels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inchannels)
        self.relu = nn.ReLU(inplace=True)
        self.shorcut = nn.Sequential(
                nn.Conv2d(inchannels, inchannels, kernel_size=1, stride=2, bias=False),
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

# Decoder with Squeeze-Excitation Module
class SEDecoderLayer(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, stride=1):
        super().__init__()

        self.shortcut = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels))
        
        
        self.residual_layer = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels // 4, 1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels))

        # Post Upscale Conv for Channel 1
        self.in_channel1_conv = nn.Sequential(
            nn.Conv2d(in_channels1, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        # Conv for Channel 2
        self.in_channel2_conv = nn.Sequential(
            nn.Conv2d(in_channels2, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        # SE Module
        self.se_layer = nn.Sequential(*list(SqueezeExcitation(out_channels, out_channels // 16).children()))
        


    def forward(self, x, encoder_x):
        x = F.upsample(x, size=encoder_x.size()[2:], mode='bilinear')
        
        x = self.in_channel1_conv(x)

        se_out = self.se_layer(x)

        encoder_x = self.in_channel2_conv(encoder_x)
        encoder_x = se_out * encoder_x
        
        x = torch.cat([x, encoder_x], dim=1)
        shortcut = self.shortcut(x)
        residual = self.residual_layer(x)
        return F.relu(shortcut + residual)


# Inner-Connected Module for segmentation logits and detection
class InnerConnectedModule(nn.Module):
    def __init__(self, in_channels, n_boxes, n_classes, image_size, stride=1, disable_det=False):
        super().__init__()
        
        self.image_size = image_size
        self.n_classes = n_classes
        self.n_boxes = n_boxes
        self.disable_det = disable_det

        # Conv for seg output
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels, n_classes + 1, kernel_size=3, stride=stride, padding=1)
        )

        if not disable_det:
            # Conv before concat with skip connection
            self.conv1 = nn.Sequential(
                nn.Conv2d(n_classes + 1, 48, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.Conv2d(48, 128, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True))

            # Conv after concat with skip connection
            self.conv2 = nn.Sequential(
                nn.Conv2d(128+in_channels, 256, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True))

            # Conv for loc and conf outputs
            self.loc_head = nn.Sequential(
                nn.Conv2d(256, n_boxes * 4, 3, padding=1))
            self.conf_head = nn.Sequential(
                nn.Conv2d(256, n_boxes * (n_classes + 1), 3, padding=1))


    def forward(self, x):
        shortcut = x
        
        x = self.seg_head(x)
        seg_out = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=True)

        if self.disable_det:
            return seg_out

        x = self.conv1(x)
        x = torch.cat([x, shortcut], dim=1)
        x = self.conv2(x)

        conf_out = self.conf_head(x)
        conf_out = conf_out.permute(0, 2, 3, 1).contiguous().view(conf_out.size(0), -1, self.n_classes + 1)

        loc_out = self.loc_head(x)
        loc_out = loc_out.permute(0, 2, 3, 1).contiguous().view(loc_out.size(0), -1, 4)
        
        return seg_out, conf_out, loc_out

class PairNet(nn.Module):
    def __init__(self, n_classes, aspect_ratios, image_size=300):
        super().__init__()
        self.n_classes = n_classes
        self.image_size = image_size

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

        self.skip_layers = ['res2', 'res3', 'res4', 'res5', 'res6','res7']

        self._initialize_weights(self.res5_7)

        self.encoder = nn.Sequential(self.res1_4, self.res5_7)

        self.last_encoder_conv = nn.Conv2d(2048, 512, 1, bias=False)

        self.decoder = nn.Sequential(OrderedDict([
            ('decoder1', DecoderLayer(2048, 2048, 512)),
            ('decoder2', DecoderLayer(512,  2048, 512)),
            ('decoder3', DecoderLayer(512,  2048, 512)),
            ('decoder4', DecoderLayer(512,  1024, 512)),
            ('decoder5', DecoderLayer(512,  512,  512))]
        ))

        n_decoder_output = len(self.decoder._modules.items()) + 1

        self.list_segmentation_head_head = nn.ModuleList([])
        for i in range(n_decoder_output):
            self.list_segmentation_head_head.append(nn.Conv2d(512, self.n_classes + 1, kernel_size=3, stride=1, padding=1))

    def _initialize_weights(self, block):
        for module in block.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
    
    def forward(self, x, is_eval=False):
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

        list_decoder_embedding = [self.last_encoder_conv(list_encoder_embedding[0])]
        list_encoder_embedding = list_encoder_embedding[1:]
        for i, (name, m) in enumerate(self.decoder._modules.items()):
            x = m(x, list_encoder_embedding[i])
            list_decoder_embedding.append(x)

        return self.segmentation_prediction(list_decoder_embedding, is_eval)

    def segmentation_prediction(self, xs, is_eval):
        list_seg_hat = []
        if is_eval:
            x = xs[-1]
            out = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=True)
            out = self.list_segmentation_head_head[-1](out)
            return out
        else:
            for i, x in enumerate(xs):
                out = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=True)
                out = self.list_segmentation_head_head[i](out)
                list_seg_hat.append(out)
            return list_seg_hat


class TripleNet(nn.Module):
    def __init__(self, n_classes, aspect_ratios, image_size=300, disable_det=False):
        super().__init__()
        self.n_classes = n_classes
        self.image_size = image_size
        self.disable_det = disable_det

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
        
        self.skip_layers = ['res2', 'res3', 'res4', 'res5', 'res6','res7']

        self._initialize_weights(self.res5_7)

        self.encoder = nn.Sequential(self.res1_4, self.res5_7)

        self.last_encoder_conv = nn.Conv2d(2048, 512, 1, bias=False)

        # Decode layers with Squeeze-Excitation Module
        # (Also known as attention skip-layer fusion)
        self.decoder = nn.Sequential(OrderedDict([
            ('decoder1', SEDecoderLayer(2048, 2048, 512)),
            ('decoder2', SEDecoderLayer(512, 2048, 512)),
            ('decoder3', SEDecoderLayer(512,  2048, 512)),
            ('decoder4', SEDecoderLayer(512,  1024, 512)),
            ('decoder5', SEDecoderLayer(512,  512, 512))]
        ))

        n_boxes = len(aspect_ratios) + 1
        n_decoder_output = len(self.decoder._modules.items()) + 1

        # Inner Connected Module
        self.list_inner_conn_modules = nn.ModuleList([])
        for i in range(n_decoder_output):
            self.list_inner_conn_modules.append(
                InnerConnectedModule(512, n_boxes, self.n_classes, self.image_size, disable_det=self.disable_det))

        # Multi-scale Fused Segmentation
        self.msf_seg_head = nn.Sequential(
            nn.Conv2d(512*n_decoder_output, self.n_classes + 1, kernel_size=3, stride=1, padding=1)
        )

        # Class-agnostic segmentation
        self.clsag_seg_head = nn.Sequential(
            nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
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
    
    def forward(self, x, is_eval=False):
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

        list_decoder_embedding = [self.last_encoder_conv(list_encoder_embedding[0])]
        list_encoder_embedding = list_encoder_embedding[1:]

        for i, (name, m) in enumerate(self.decoder._modules.items()):
            x = m(x, list_encoder_embedding[i])
            list_decoder_embedding.append(x)

        # Multi-scale fused Segmentation
        seg_hat_msf = self.msf_seg_prediction(list_decoder_embedding)
        
        if is_eval:
            # Evaluation
            if not self.disable_det:
                # Joint Det and Seg Eval

                # Inner Connected Module Output
                locs = []
                confs = []
                for i, decoder_embedding in enumerate(list_decoder_embedding):
                    seg_out, conf_out, loc_out = self.list_inner_conn_modules[i](decoder_embedding)
                    locs.append(loc_out)
                    confs.append(conf_out)

                loc_hat = torch.cat(locs, dim=1)
                conf_hat = torch.cat(confs, dim=1)

                return loc_hat, conf_hat, seg_hat_msf
            else:
                # Seg Only Eval
                return seg_hat_msf
        else:
            #Training

            # Class-agnostic segmentation
            list_seg_hat_clsag = self.class_agnos_seg_prediction(list_decoder_embedding, is_eval)

            if not self.disable_det:
                # Joint Det and Seg Training

                # Inner Connected Module Output
                locs = []
                confs = []
                list_seg_hat = []
                for i, decoder_embedding in enumerate(list_decoder_embedding):
                    seg_out, conf_out, loc_out = self.list_inner_conn_modules[i](decoder_embedding)
                    locs.append(loc_out)
                    confs.append(conf_out)
                    list_seg_hat.append(seg_out)

                loc_hat = torch.cat(locs, dim=1)
                conf_hat = torch.cat(confs, dim=1)

                return loc_hat, conf_hat, list_seg_hat, seg_hat_msf, list_seg_hat_clsag
            else:
                # Seg Only Train

                # Inner Connected Module Output
                list_seg_hat = []
                for i, decoder_embedding in enumerate(list_decoder_embedding):
                    seg_out = self.list_inner_conn_modules[i](decoder_embedding)
                    list_seg_hat.append(seg_out)

                return list_seg_hat, seg_hat_msf, list_seg_hat_clsag

            



    # Multi-scale Fused Segmentation 
    def msf_seg_prediction(self, xs):
        list_maps = []
        for x in xs:
            upsampled_map = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=True)
            list_maps.append(upsampled_map)
        
        concated_map = torch.cat(list_maps, dim=1)
        msf_seg_hat = self.msf_seg_head(concated_map)
        return msf_seg_hat

    # Class-agnostic segmentation
    def class_agnos_seg_prediction(self, xs, is_eval):
        
        if is_eval:
            # Evaluation
            x = xs[-1]
            out = self.clsag_seg_head(x)
            seg_hat_cls_agnos = F.interpolate(out, size=self.image_size, mode='bilinear', align_corners=True)

            return seg_hat_cls_agnos
        else:
            # Training
            list_seg_hat_cls_agnos = []
            for i, x in enumerate(xs):
                out = self.clsag_seg_head(x)
                out = F.interpolate(out, size=self.image_size, mode='bilinear', align_corners=True)
                list_seg_hat_cls_agnos.append(out)
            
            return list_seg_hat_cls_agnos

