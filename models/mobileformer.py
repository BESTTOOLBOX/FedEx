import torch
import torch.nn as nn

from torch.nn import init
import torch.nn.functional as F

from models.mobile import Mobile, MobileDown
from models.former import Former
from models.bridge import Mobile2Former, Former2Mobile
#from mobile import Mobile, MobileDown
#from former import Former
#from bridge import Mobile2Former, Former2Mobile

class BaseBlock(nn.Module):
    def __init__(self, inp, exp, out, se, stride, heads, dim):
        super(BaseBlock, self).__init__()
        if stride == 2:
            self.mobile = MobileDown(3, inp, exp, out, stride, dim)
        else:
            self.mobile = Mobile(3, inp, exp, out, stride, dim)
        self.mobile2former = Mobile2Former(dim=dim, heads=heads, channel=inp)
        self.former = Former(dim=dim)
        self.former2mobile = Former2Mobile(dim=dim, heads=heads, channel=out)

    def forward(self, x, z):
        z_hid = self.mobile2former(x, z)
        z_out = self.former(z_hid)
        x_hid = self.mobile(x, z_out)
        x_out = self.former2mobile(x_hid, z_out)
        return x_out, z_out


class MobileFormer(nn.Module):
    def __init__(self, cfg):
        super(MobileFormer, self).__init__()
        # 初始化6*192token
        self.token = nn.Parameter(nn.Parameter(torch.randn(1, cfg['token'], cfg['embed'])))
        # stem 3 224 224 -> 16 112 112
        self.stem = nn.Sequential(
            nn.Conv2d(3, cfg['stem'], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(cfg['stem']),
            nn.Hardswish(),
        )
        # bneck 先*2后还原，步长为1，组卷积
        self.bneck = nn.Sequential(
            nn.Conv2d(cfg['stem'], cfg['bneck']['e'], kernel_size=3, stride=cfg['bneck']['s'], padding=1, groups=cfg['stem']),
            nn.Hardswish(),
            nn.Conv2d(cfg['bneck']['e'], cfg['bneck']['o'], kernel_size=1, stride=1),
            nn.BatchNorm2d(cfg['bneck']['o'])
        )

        # body
        self.block = nn.ModuleList()
        for kwargs in cfg['body']:
            # 把{'inp': 12, 'exp': 72, 'out': 16, 'se': None, 'stride': 2, 'heads': 2}和token维度192传进去
            self.block.append(BaseBlock(**kwargs, dim=cfg['embed']))

        inp = cfg['body'][-1]['out']
        exp = cfg['body'][-1]['exp']
        self.conv = nn.Conv2d(inp, exp, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(exp)
        self.avg = nn.AvgPool2d((7, 7))

        self.head = nn.Sequential(
            nn.Linear(exp + cfg['embed'], cfg['fc1']),
            nn.Hardswish(),
            nn.Linear(cfg['fc1'], cfg['fc2'])
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # batch_size
        b = x.shape[0]
        # 因为最开始初始化的是1*6*192，在0维度重复b次，1维度重复1次，2维度重复1次，就形成了b*6*192
        z = self.token.repeat(b, 1, 1)# batchsize * 6 * 192

        x = self.bneck(self.stem(x))# batchsize * 12 * 28 * 28

        for m in self.block:
            x, z = m(x, z)

        #print('after block x:', x.shape)# batchsize * 128 * 7 * 7
        #print('after block z:', z.shape)
        # 转成b个平铺一维向量
        x = self.avg(self.bn(self.conv(x))).view(b, -1)# batchsize * 768
        
        # 取第一个token
        z = z[:, 0, :].view(b, -1)# batchsize * 192

        # 最后一个维度拼接
        out = torch.cat((x, z), -1)# batchsize * 960

        return self.head(out)#batchsize * 58


if __name__ == '__main__':
    cfg = {
        'name': 'mf151',
        'token': 6,  # tokens
        'embed': 192,  # embed_dim
        'stem': 12,
        # stage1
        'bneck': {'e': 24, 'o': 12, 's': 1},  # exp out stride
        'body': [
            # stage2
            {'inp': 12, 'exp': 72, 'out': 16, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 16, 'exp': 48, 'out': 16, 'se': None, 'stride': 1, 'heads': 2},
            # stage3
            {'inp': 16, 'exp': 96, 'out': 32, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 32, 'exp': 96, 'out': 32, 'se': None, 'stride': 1, 'heads': 2},
            # stage4
            {'inp': 32, 'exp': 192, 'out': 64, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 64, 'exp': 256, 'out': 64, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 64, 'exp': 384, 'out': 88, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 88, 'exp': 528, 'out': 88, 'se': None, 'stride': 1, 'heads': 2},
            # stage5
            {'inp': 88, 'exp': 528, 'out': 128, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 128, 'exp': 768, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 128, 'exp': 768, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
        ],
        'fc1': 1280,  # hid_layer
        'fc2': 58   # num_classes
    }
    inputs = torch.randn(1, 3, 28, 28)
    model = MobileFormer(cfg)
    outputs = model(inputs)
    print(outputs.shape)