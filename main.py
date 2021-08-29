# VAE as GAN's Generator | GAN-Discriminator as VAE's Loss Measurement
# By Mo Wentao @MatthewL
# This code is based on github.com/seangal/dcgan_vae_pytorch and is referred to
# github.com/zhangqianhui/vae-gan-tensorflow/, while both implementation   
# is somewhat broken and not correct, and the original paper is arxiv.org/abs/1512.09300


import argparse
import os
import random
import math
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision
from torch.utils.tensorboard import SummaryWriter
# from torch.autograd import Variable

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--saveInt', type=int, default=25, help='number of epochs between checkpoints')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--optim', help='optimizer used')
    parser.add_argument('--post_fix', help='tensorboard postfix')
    parser.add_argument('--delayD', type=int, default=0, help='delay updates of D')
    parser.add_argument('--decay', type=float, default=0.99, help='exponential decay factor')
    parser.add_argument('--resnetD', action='store_true', help='use resNet on discriminator')
    parser.add_argument('--pretrained', action='store_true', help='requires --resnetD, load a pretrained resnet and finetune')

    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=opt.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(opt.imageSize),
                                       transforms.CenterCrop(opt.imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    elif opt.dataset == 'lsun':
        dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ])
                               )
    assert dataset, "Failed"
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, drop_last=True,
                                             shuffle=True, num_workers=int(opt.workers), pin_memory=opt.cuda)

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = 3

    # extract number
    def to_digits(s):
        s = s[-10:]
        res = ''
        for c in s:
            if '0' <= c <= '9':
                res += c
        return int(res)

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def set_parameters_freeze(model, freeze=True):
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        else:   # unfreeze
            for param in model.parameters():
                param.requires_grad = True

    class _Sampler(nn.Module):
        def __init__(self):
            super(_Sampler, self).__init__()

        def forward(self, input):
            mu = input[0]
            logvar = input[1]

            std = logvar.mul(0.5).exp_()  # calculate the STDEV
            if opt.cuda:
                eps = torch.cuda.FloatTensor(std.size()).normal_()  # random normalized noise
            else:
                eps = torch.FloatTensor(std.size()).normal_()  # random normalized noise
            # eps = Variable(eps)
            return eps.mul(std).add_(mu)


    class _Encoder(nn.Module):
        def __init__(self, imageSize):
            super(_Encoder, self).__init__()

            n = math.log2(imageSize)

            assert n == round(n), 'imageSize must be a power of 2'
            assert n >= 3, 'imageSize must be at least 8'
            n = int(n)

            self.conv1 = nn.Conv2d(ngf * 2 ** (n - 3), nz, 4)
            self.conv2 = nn.Conv2d(ngf * 2 ** (n - 3), nz, 4)

            self.encoder = nn.Sequential()
            # input is (nc) x 64 x 64
            self.encoder.add_module('input-conv', nn.Conv2d(nc, ngf, 4, 2, 1, bias=False))
            self.encoder.add_module('input-relu', nn.LeakyReLU(0.2, inplace=True))
            for i in range(n - 3):
                # state size. (ngf) x 32 x 32
                self.encoder.add_module('pyramid-{0}-{1}-conv'.format(ngf * 2 ** i, ngf * 2 ** (i + 1)),
                                        nn.Conv2d(ngf * 2 ** (i), ngf * 2 ** (i + 1), 4, 2, 1, bias=False))
                self.encoder.add_module('pyramid-{0}-batchnorm'.format(ngf * 2 ** (i + 1)),
                                        nn.BatchNorm2d(ngf * 2 ** (i + 1)))
                self.encoder.add_module('pyramid-{0}-relu'.format(ngf * 2 ** (i + 1)), nn.LeakyReLU(0.2, inplace=True))

            # state size. (ngf*8) x 4 x 4

        def forward(self, input):
            output = self.encoder(input)
            return [self.conv1(output), self.conv2(output)]


    class _netG(nn.Module):
        def __init__(self, imageSize, ngpu):
            super(_netG, self).__init__()
            self.ngpu = ngpu
            self.encoder = _Encoder(imageSize)
            self.sampler = _Sampler()

            n = math.log2(imageSize)

            assert n == round(n), 'imageSize must be a power of 2'
            assert n >= 3, 'imageSize must be at least 8'
            n = int(n)

            self.decoder = nn.Sequential()
            # input is Z, going into a convolution
            self.decoder.add_module('input-conv', nn.ConvTranspose2d(nz, ngf * 2 ** (n - 3), 4, 1, 0, bias=False))
            self.decoder.add_module('input-batchnorm', nn.BatchNorm2d(ngf * 2 ** (n - 3)))
            self.decoder.add_module('input-relu', nn.LeakyReLU(0.2, inplace=True))

            # state size. (ngf * 2**(n-3)) x 4 x 4

            for i in range(n - 3, 0, -1):
                self.decoder.add_module('pyramid-{0}-{1}-conv'.format(ngf * 2 ** i, ngf * 2 ** (i - 1)),
                                        nn.ConvTranspose2d(ngf * 2 ** i, ngf * 2 ** (i - 1), 4, 2, 1, bias=False))
                self.decoder.add_module('pyramid-{0}-batchnorm'.format(ngf * 2 ** (i - 1)),
                                        nn.BatchNorm2d(ngf * 2 ** (i - 1)))
                self.decoder.add_module('pyramid-{0}-relu'.format(ngf * 2 ** (i - 1)), nn.LeakyReLU(0.2, inplace=True))

            self.decoder.add_module('ouput-conv', nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False))
            self.decoder.add_module('output-tanh', nn.Tanh())

        def forward(self, input):
            if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.encoder, input, range(self.ngpu))
                output = nn.parallel.data_parallel(self.sampler, output, range(self.ngpu))
                output = nn.parallel.data_parallel(self.decoder, output, range(self.ngpu))
            else:
                output = self.encoder(input)
                output = self.sampler(output)
                output = self.decoder(output)
            return output

        def make_cuda(self):
            self.encoder.cuda()
            self.sampler.cuda()
            self.decoder.cuda()


    netG = _netG(opt.imageSize, ngpu)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))


    class _netD(nn.Module):
        def __init__(self, imageSize, ngpu, useResnet=False, pretrained=True):
            super(_netD, self).__init__()
            self.ngpu = ngpu
            n = math.log2(imageSize)

            assert n == round(n), 'imageSize must be a power of 2'
            assert n >= 3, 'imageSize must be at least 8'
            n = int(n)

            if useResnet:
                self.main = torchvision.models.resnet50(pretrained=pretrained, progress=True)
                # self.main = torchvision.models.squeezenet1_1(pretrained=pretrained, progress=False)
                # self.main = torchvision.models.densenet121(pretrained=False, progress=True)
                self.main.train()
                num_ftrs = self.main.fc.in_features
                # notice: experiments proved that resNet/denseNet/squeezeNet can hardly train Generator
                # since their convergence speed and accuracy is too high, that Generator can't
                # learn a way to improve its results.
                if pretrained: # finetune
                    set_parameters_freeze(self.main)
                    # enhance FC if fine-tune
                    self.main.fc = nn.Sequential(nn.Linear(num_ftrs, 64),
                                                 nn.LeakyReLU(0.2, inplace=True),
                                                 nn.Linear(64, 1),
                                                 nn.Sigmoid())
                else:
                    # pass
                    self.main.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())

            else:
                self.main = nn.Sequential()

                # input is (nc) x 64 x 64
                self.main.add_module('input-conv', nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
                self.main.add_module('relu', nn.LeakyReLU(0.2, inplace=True))

                # state size. (ndf) x 32 x 32
                for i in range(n - 3):
                    self.main.add_module('pyramid-{0}-{1}-conv'.format(ndf * 2 ** (i), ndf * 2 ** (i + 1)),
                                         nn.Conv2d(ndf * 2 ** (i), ndf * 2 ** (i + 1), 4, 2, 1, bias=False))
                    self.main.add_module('pyramid-{0}-batchnorm'.format(ndf * 2 ** (i + 1)),
                                         nn.BatchNorm2d(ndf * 2 ** (i + 1)))
                    self.main.add_module('pyramid-{0}-relu'.format(ndf * 2 ** (i + 1)), nn.LeakyReLU(0.2, inplace=True))

                self.main.add_module('output-conv', nn.Conv2d(ndf * 2 ** (n - 3), 1, 4, 1, 0, bias=False))
                self.main.add_module('output-sigmoid', nn.Sigmoid())

        def forward(self, input):
            if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)
            if opt.resnetD:
                output = self.fc(output)
                # print(output.size())
            # put.size())
            return output.view(-1, 1)


    netD = _netD(opt.imageSize, ngpu, useResnet=opt.resnetD, pretrained=opt.pretrained)
    netD.apply(weights_init)

    beg_epoch = 0
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
        beg_epoch = to_digits(opt.netD)

    BCECriterion = nn.BCELoss()
    MSECriterion = nn.MSELoss()

    real_label = 1
    fake_label = 0
    input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    # fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    labelR = torch.FloatTensor(opt.batchSize, 1).fill_(real_label)
    labelF = torch.FloatTensor(opt.batchSize, 1).fill_(fake_label)

    if opt.cuda:
        netD.cuda()
        netG.make_cuda()
        BCECriterion.cuda()
        MSECriterion.cuda()
        input, labelF, labelR = input.cuda(), labelF.cuda(), labelR.cuda()
        noise = noise.cuda()
        # fixed_noise = fixed_noise.cuda()

    # setup optimizer
    if opt.optim == 'Adam': # appliable
        optimizerD = optim.Adam([{'params': netD.parameters(), 'initial_lr': opt.lr}], lr=opt.lr,
                                betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam([{'params': netG.parameters(), 'initial_lr': opt.lr}], lr=opt.lr,
                                betas=(opt.beta1, 0.999))
    elif opt.optim == 'AdamW':
        optimizerD = optim.AdamW([{'params': netD.parameters(), 'initial_lr': opt.lr}], lr=opt.lr, betas=(opt.beta1, 0.999))
        optimizerG = optim.AdamW([{'params': netG.parameters(), 'initial_lr': opt.lr}], lr=opt.lr, betas=(opt.beta1, 0.999))
    elif opt.optim == 'SGD': # very slow/unstable
        optimizerD = optim.SGD([{'params': netD.parameters(), 'initial_lr': opt.lr}],
                               lr=opt.lr)  # , momentum=1e-3, nesterov=True)
        optimizerG = optim.SGD([{'params': netG.parameters(), 'initial_lr': opt.lr}],
                               lr=opt.lr)  # ,momentum=1e-3, nesterov=True)
    elif opt.optim == 'RMSprop': # less appliable
        optimizerD = optim.RMSprop([{'params': netD.parameters(), 'initial_lr': opt.lr}],
                                   lr=opt.lr)  # , momentum=1e-3, nesterov=True)
        optimizerG = optim.RMSprop([{'params': netG.parameters(), 'initial_lr': opt.lr}],
                                   lr=opt.lr)  # ,momentum=1e-3, nesterov=True)
    elif opt.optim == 'NSGD': # even more unstable
        optimizerD = optim.SGD([{'params': netD.parameters(), 'initial_lr': opt.lr}],
                               lr=opt.lr, momentum=1e-3, nesterov=True)
        optimizerG = optim.SGD([{'params': netG.parameters(), 'initial_lr': opt.lr}],
                               lr=opt.lr, momentum=1e-3, nesterov=True)

    assert optimizerD, "Unsupported optimizer type"
    assert optimizerG, "Unsupported optimizer type"

    # scheduler
    schedulerD = optim.lr_scheduler.ExponentialLR(optimizerD, gamma=opt.decay, last_epoch=beg_epoch)
    schedulerG = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=opt.decay, last_epoch=beg_epoch)

    post_fix = opt.post_fix
    writer = SummaryWriter(comment=opt.outf + post_fix)
    print("Resuming from Epoch: ", beg_epoch)
    # beg_epoch=35
    global_step = beg_epoch * len(dataloader)
    start_time = time.time()

    for epoch in range(beg_epoch + 1, opt.niter):
        batch_time = time.time()
        for i, data in enumerate(dataloader, 0):
            global_step += 1
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) + log(1 - D(G(Enc(x)))
            ###########################
            # train with real
            input, _ = data
            batch_size = input.size(0)

            if opt.cuda:
                input = input.cuda()
            #
            output = netD(input)
            errD_real = BCECriterion(output, labelR)
            # errD_real.backward()
            D_x = output.data.mean()

            # train with fake
            noise.resize_(batch_size, nz, 1, 1)
            noise.normal_(0, 1)
            gen = netG.decoder(noise)

            output = netD(gen.detach())
            errD_fake = BCECriterion(output, labelF)
            D_G_z1 = output.data.mean()

            # train with rebuild
            rec = netG(input)
            output = netD(rec)
            errD_rebuild = BCECriterion(output, labelF)
            errD = errD_real + errD_fake + errD_rebuild
            errD.backward()

            # update for each 10 batches
            if opt.delayD != 0:
                if i % opt.delayD == 0 and i != 0:
                    optimizerD.step()
                    netD.zero_grad()
            else:
                optimizerD.step()
                netD.zero_grad()

            ############################
            # (2) Update G network: VAE
            ###########################
            encoded = netG.encoder(input)
            mu = encoded[0]
            logvar = encoded[1]

            KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            KLD = torch.sum(KLD_element).mul_(-0.5)
            sampled = netG.sampler(encoded)
            rec = netG.decoder(sampled)  # rebuild
            MSEerr = MSECriterion(rec, input)
            VAEerr = 100. * KLD / (batch_size * nz) + MSEerr

            ############################
            # (3) Update G network: maximize log(D(G(z)))
            ###########################
            output = netD(rec)  # rebuild label
            errG_rebuild = BCECriterion(output, labelR)
            noise.normal_()
            gen = netG.decoder(noise)  # generated
            output_gen = netD(gen)
            errG_gen = BCECriterion(output_gen, labelR)
            # coefficient = 3.0 if opt.resnetD else 1.0 # if using resnet, boost Generator
            errG = VAEerr + (errG_rebuild + errG_gen)
            errG.backward()
            D_G_z2 = output.data.mean()  # rebuild average label
            # update every 10 batches
            # if True or (i % batch_update_period == 0 and i != 0):
            optimizerG.step()
            netG.zero_grad()

            # send to tensor-board every 5 steps
            if global_step % 5 == 0 and global_step != 0:
                writer.add_image('generated', gen.data[0].cpu() * 0.5 + 0.5, global_step)
                writer.add_image('rebuild', rec.data[0].cpu() * 0.5 + 0.5, global_step)
                writer.add_image('original', input.data[0].cpu() * 0.5 + 0.5, global_step)
                vutils.save_image(gen.data[0].cpu() * 0.5 + 0.5, "%s/%s-%d-%d.gen.png" % (opt.outf, post_fix, epoch, i))
                vutils.save_image(rec.data[0].cpu() * 0.5 + 0.5,
                                  "%s/r%s-%d-%d.reb.png" % (opt.outf, post_fix, epoch, i))
                vutils.save_image(input.data[0].cpu() * 0.5 + 0.5,
                                  "%s/o%s-%d-%d.ori.png" % (opt.outf, post_fix, epoch, i))
                record_dict = {
                    "loss_vae": VAEerr.item(),
                    "loss_d": errD.item(),
                    "loss_g": errG.item(),
                    "label_x": D_x.data.item(),
                    "label_generated": D_G_z1.data.item(),
                    "label_rebuild": D_G_z2.data.item()
                }

                for key in record_dict:
                    writer.add_scalar(key, record_dict[key], global_step)

            current = time.time()

            # inform console
            print('[%d/%d][%d/%d]Loss_VAE:%.4f Loss_D:%.4f Loss_G:%.4f D(x):%.4f D(G(z)):%.4f/%.4f Time:%.2f/%.2f'
                  % (epoch, opt.niter, i, len(dataloader),
                     VAEerr.item(), errD.item(), errG_rebuild.item(),
                     D_x, D_G_z1, D_G_z2,
                     (current - batch_time) / (i + 1), current - start_time
                     )
            )

        schedulerD.step()
        schedulerG.step()
        if epoch % opt.saveInt == 0 and epoch != 0:
            torch.save(netG.state_dict(), '%s/%s-netG_epoch_%d.pth' % (opt.outf, post_fix, epoch))
            torch.save(netD.state_dict(), '%s/%s-netD_epoch_%d.pth' % (opt.outf, post_fix, epoch))
