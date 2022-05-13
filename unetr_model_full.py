from .unetr_parts import *


class UNetRFull(nn.Module):
    def __init__(self, n_channels, n_classes=1, model_parallelism=False, args=''):

        device_count = torch.cuda.device_count()

        self.cuda0 = torch.device('cuda:0')
        self.cuda1 = torch.device('cuda:0')
        self.model_parallelism = model_parallelism
        if self.model_parallelism:
            if device_count > 1:
                self.cuda1 = torch.device('cuda:1')
                print('Using Model Parallelism with 2 gpu')
            else:
                print('Can not use model parallelism! Only found 1 GPU device!')
                self.cuda1 = torch.device('cuda:0')

        super(UNetRFull, self).__init__()

        input_feature_len = len(args.input_feature.split(sep=',')) - 1
        if args.use_sagital:
            n_channels += 1

        n_filter = [8, 16, 32, 64, 128]

        self.inc = inconv(n_channels, n_filter[0]).cuda(self.cuda0)
        self.down1 = down(n_filter[0], n_filter[1]).cuda(self.cuda0)
        self.down2 = down(n_filter[1], n_filter[2]).cuda(self.cuda0)
        self.down3 = down(n_filter[2], n_filter[3]).cuda(self.cuda0)
        self.down4 = down(n_filter[3], n_filter[3]).cuda(self.cuda0)
        self.up1 = up(n_filter[4], n_filter[2]).cuda(self.cuda1)
        self.up2 = up(n_filter[3], n_filter[1]).cuda(self.cuda1)
        self.up3 = up(n_filter[2], n_filter[0]).cuda(self.cuda1)
        self.up4 = up(n_filter[1], n_filter[0]).cuda(self.cuda1)
        self.outc = outconv(n_filter[0], 1).cuda(self.cuda1)
        self.net_linear = torch.nn.Sequential(
            torch.nn.Linear(2048*2048 + input_feature_len, 50),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(50, 10),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(10, n_classes),
        ).cuda(self.cuda1)

    def forward(self, x, input_feat):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.model_parallelism:
            x1 = x1.cuda(self.cuda1)
            x2 = x2.cuda(self.cuda1)
            x3 = x3.cuda(self.cuda1)
            x4 = x4.cuda(self.cuda1)
            x5 = x5.cuda(self.cuda1)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = x.view(-1, 2048 * 2048)
        xs = torch.cat((x, input_feat.cuda(self.cuda1)), 1)
        reg_output = self.net_linear(xs).cuda(self.cuda0)
        
        return reg_output
