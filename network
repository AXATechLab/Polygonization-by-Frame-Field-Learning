import math
import torch
from torch import nn
from torch.nn import functional as F

#All of the classes except the last two are taken from https://github.com/bohaohuang/mrs

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNet(nn.Module):
    def __init__(self, block, layers=(3, 4, 6, 3), strides=(2, 2, 2, 2, 2), inter_features=True, groups=1,
                 width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 64
        self.inter_features = inter_features
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=strides[0], padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=strides[1], padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[2])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[3], dilation=2//strides[3])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[4], dilation=4**(2-strides[4]))
        self.chans = [64, 64*block.expansion, 128*block.expansion, 256*block.expansion, 512*block.expansion][::-1]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        base_width=self.base_width, norm_layer=norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, dilation=dilation,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.inter_features:
            x = self.conv1(x)
            x = self.bn1(x)
            layer0 = self.relu(x)
            x = self.maxpool(layer0)

            layer1 = self.layer1(x)
            layer2 = self.layer2(layer1)
            layer3 = self.layer3(layer2)
            layer4 = self.layer4(layer3)

            return layer4, layer3, layer2, layer1, layer0
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            return x

class Base(nn.Module):
    def __init__(self):
        self.lbl_margin = 0
        super(Base, self).__init__()

    def forward(self, *inputs_):
        """
        Forward operation in network training
        This does not necessarily equals to the inference, i.e., less output in inference
        :param inputs_:
        :return:
        """
        raise NotImplementedError

    def inference(self, *inputs_):
        outputs = self.forward(*inputs_)['pred']
        if isinstance(outputs, tuple):
            return outputs[0]
        else:
            return outputs

    def init_weight(self):
        """
        Initialize weights of the model
        :return:
        """
        for m in network_utils.iterate_sublayers(self):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform(m.weight)
                torch.nn.init.xavier_uniform(m.bias)

class UpSample(nn.Module):
    """
    This module defines the upsample operation in the D-LinkNet
    """
    def __init__(self, in_chan, out_chan):
        super(UpSample, self).__init__()
        self.chan = in_chan
        self.conv1 = nn.Conv2d(in_chan, in_chan//4, kernel_size=1, stride=1, padding=0)
        self.tconv = nn.ConvTranspose2d(in_chan//4, in_chan//4, kernel_size=(3, 3), stride=2, padding=1, \
                                        output_padding=1)
        self.conv2 = nn.Conv2d(in_chan//4, out_chan, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tconv(x)
        x = self.conv2(x)
        return x

class CenterDilation(nn.Module):
    """
    This module defines the center dilation part of the D-Link Net
    """
    def __init__(self, chan):
        super(CenterDilation, self).__init__()
        self.chan = chan
        self.dconv_1 = nn.Conv2d(chan, chan, 3, 1, 1, 1)
        self.dconv_2 = nn.Conv2d(chan, chan, 3, 1, 2, 2)
        self.dconv_3 = nn.Conv2d(chan, chan, 3, 1, 4, 4)
        self.dconv_4 = nn.Conv2d(chan, chan, 3, 1, 8, 8)

    def forward(self, x):
        x_1 = self.dconv_1(x)
        x_2 = self.dconv_2(x_1)
        x_3 = self.dconv_3(x_2)
        x_4 = self.dconv_4(x_3)
        x = x + x_1 + x_2 + x_3 + x_4
        return x

class DLinkNetDecoder(Base):
    """
    This module defines part (b) and (c) in the D-LinkNet paper
    Grouping them together to match the MRS naming convention
    """
    def __init__(self, chans, n_class, final_upsample=True):
        super(DLinkNetDecoder, self).__init__()
        self.chans = chans
        self.center_dilation = CenterDilation(self.chans[0])
        self.upsample_1 = UpSample(self.chans[0], self.chans[1])
        self.upsample_2 = UpSample(self.chans[1], self.chans[2])
        self.upsample_3 = UpSample(self.chans[2], self.chans[3])
        self.upsample_4 = UpSample(self.chans[3], self.chans[4])
        if final_upsample:
            self.tconv = nn.ConvTranspose2d(self.chans[4], self.chans[4]//2, 4, 2, 1)
        else:
            self.tconv = nn.Conv2d(self.chans[4], self.chans[4]//2, 3, 1, 1)
        self.classify = nn.Conv2d(self.chans[4]//2, n_class, 3, 1, 1)

    def forward(self, ftr, layers, input_size):
        ftr = self.center_dilation(ftr)
        ftr = self.upsample_1(ftr)
        ftr = ftr + layers[0]
        ftr = self.upsample_2(ftr)
        ftr = ftr + layers[1]
        ftr = self.upsample_3(ftr)
        ftr = ftr + layers[2]
        ftr = self.upsample_4(ftr)
        ftr = self.tconv(ftr)
        return self.classify(ftr)

class DLinkNet(Base):
    """
    This module is the original DLinknet defined in paper
    http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/
    w4/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.pdf
    """
    def __init__(self, n_class, encoder, aux_loss=False, use_emau=False, use_ocr=False):
        super(DLinkNet, self).__init__()
        self.n_class = n_class
        self.aux_loss = aux_loss
        self.use_emau = use_emau
        self.use_ocr = use_ocr
        self.encoder = encoder
        if self.use_emau:
            if isinstance(self.use_emau, int):
                c = self.use_emau
            else:
                c = 64
            self.encoder.emau = emau.EMAU(self.encoder.chans[0], c)
        if self.use_ocr:
            self.encoder.ocr = ocr.OCRModule(self.n_class, *self.encoder.chans[:2][::-1], self.encoder.chans[0])
        if self.aux_loss:
            self.cls = nn.Sequential(
                nn.Linear(self.encoder.chans[0], 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.n_class)
            )
        else:
            self.cls = None
            
        self.decoder = DLinkNetDecoder(self.encoder.chans, n_class)

    def forward(self, x):
        input_size = x.shape[2:]
        output_dict = dict()
        # part a: encoder
        input_size = x.size()[2]
        x = self.encoder(x)
        ftr, layers = x[0], x[1:-1]
        if self.use_emau:
            ftr, output_dict['mu'] = self.encoder.emau(ftr)
        if self.use_ocr:
            region, ftr = self.encoder.ocr(layers[0], ftr)
            output_dict['region'] = F.interpolate(region, size=input_size, mode='bilinear', align_corners=False)
        if self.aux_loss:
            output_dict['aux'] = self.cls(F.adaptive_max_pool2d(input=ftr, output_size=(1, 1))\
                                     .view(-1, ftr.size(1)))
        # part b and c: center dilation + decoder
        pred = self.decoder(ftr, layers, input_size)
        return pred

#Slight modification of the network based on https://arxiv.org/abs/1709.05932 which imporves the performance by 8 IoU points
#on both crowdai and xbd datasets

class DLinkNetDecoder2(Base):
    """
    This module defines part (b) and (c) in the D-LinkNet paper
    Grouping them together to match the MRS naming convention
    """
    def __init__(self, chans, n_class1, n_class2, final_upsample=True):
        super(DLinkNetDecoder2, self).__init__()
        self.chans = chans
        self.center_dilation = CenterDilation(self.chans[0])
        self.upsample_1 = UpSample(self.chans[0], self.chans[1])
        self.upsample_2 = UpSample(self.chans[1], self.chans[2])
        self.upsample_3 = UpSample(self.chans[2], self.chans[3])
        self.upsample_4 = UpSample(self.chans[3], self.chans[4])
        self.relu = nn.ReLU(inplace=True)
        if final_upsample:
            self.tconv = nn.ConvTranspose2d(self.chans[4], self.chans[4]//2, 4, 2, 1)
        else:
            self.tconv = nn.Conv2d(self.chans[4], self.chans[4]//2, 3, 1, 1)
        self.classify1 = nn.Conv2d(self.chans[4]//2, n_class1, 3, 1, 1)
        self.classify2 = nn.Conv2d(self.chans[4]//2 + n_class1, n_class2, 3, 1, 1)

    def forward(self, ftr, layers, input_size):
        ftr = self.center_dilation(ftr)
        ftr = self.upsample_1(ftr)
        ftr = ftr + layers[0]
        ftr = self.upsample_2(ftr)
        ftr = ftr + layers[1]
        ftr = self.upsample_3(ftr)
        ftr = ftr + layers[2]
        ftr = self.upsample_4(ftr)
        ftr = self.tconv(ftr)
        pred1 = self.classify1(ftr)
        ftr = torch.cat((self.relu(pred1), ftr), dim=1)
        pred2 = self.classify2(ftr)
        return pred1, pred2

class DLinkNet2(Base):
    """
    This module is the original DLinknet defined in paper
    http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/
    w4/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.pdf
    """
    def __init__(self, n_class1, n_class2, encoder, aux_loss=False, use_emau=False, use_ocr=False):
        super(DLinkNet2, self).__init__()
        self.n_class1 = n_class1
        self.n_class2 = n_class2
        self.aux_loss = aux_loss
        self.use_emau = use_emau
        self.use_ocr = use_ocr
        self.encoder = encoder
        if self.use_emau:
            if isinstance(self.use_emau, int):
                c = self.use_emau
            else:
                c = 64
            self.encoder.emau = emau.EMAU(self.encoder.chans[0], c)
        if self.use_ocr:
            self.encoder.ocr = ocr.OCRModule(self.n_class, *self.encoder.chans[:2][::-1], self.encoder.chans[0])
        if self.aux_loss:
            self.cls = nn.Sequential(
                nn.Linear(self.encoder.chans[0], 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.n_class)
            )
        else:
            self.cls = None
            
        self.decoder = DLinkNetDecoder2(self.encoder.chans, n_class1, n_class2)

    def forward(self, x):
        input_size = x.shape[2:]
        output_dict = dict()
        # part a: encoder
        input_size = x.size()[2]
        x = self.encoder(x)
        ftr, layers = x[0], x[1:-1]
        if self.use_emau:
            ftr, output_dict['mu'] = self.encoder.emau(ftr)
        if self.use_ocr:
            region, ftr = self.encoder.ocr(layers[0], ftr)
            output_dict['region'] = F.interpolate(region, size=input_size, mode='bilinear', align_corners=False)
        if self.aux_loss:
            output_dict['aux'] = self.cls(F.adaptive_max_pool2d(input=ftr, output_size=(1, 1))\
                                     .view(-1, ftr.size(1)))
        # part b and c: center dilation + decoder
        pred1, pred2 = self.decoder(ftr, layers, input_size)
        return pred1, pred2
