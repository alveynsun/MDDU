import torch
from .base_model import BaseModel
from . import networks
from util.image_pool import ImagePool


class ShadowRemovalModel(BaseModel):


    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(norm='instance', dataset_mode='shadow')
        parser.set_defaults(input_nc=3, output_nc=3)
        parser.add_argument('--share_weights', action='store_true', help='share weights')
        parser.add_argument('--num_iterations', type=int, default=3, help='K iterations')
        parser.add_argument('--log_eps', type=float, default=1e-4, help='log epsilon')
        if is_train:

            parser.set_defaults(pool_size=50, gan_mode='lsgan')
            parser.add_argument('--use_mask', action='store_true',
                                help='use shadow mask if available')
            parser.add_argument('--lambda_physical', type=float, default=10.0,
                                help='weight for physical reconstruction loss')
            parser.add_argument('--lambda_gan', type=float, default=1.5,
                                help='weight for adversarial loss')
            parser.add_argument('--lambda_decomp', type=float, default=3.0,
                                help='weight for decomposition loss')
            parser.add_argument('--mask_weight_factor', type=float, default=2.0,
                                help='extra weight for shadow regions in physical loss')
            parser.add_argument('--decomp_temp', type=float, default=0.5,
                                help='temperature for soft valid region in decomposition loss')
            parser.add_argument('--lambda_perceptual', type=float, default=0.5,
                                help='weight for perceptual loss')
            parser.add_argument('--lambda_boundary', type=float, default=0.5,
                                help='weight for shadow boundary smoothness loss')
            parser.add_argument('--d_update_freq', type=int, default=2,
                                help='update D every N iterations')
            parser.add_argument('--label_smoothing', type=float, default=0.1,
                                help='label smoothing for real labels in D')

        return parser

    def __init__(self, opt):
        """Initialize the shadow removal model."""
        BaseModel.__init__(self, opt)

        self.loss_names = ['G_GAN', 'G_Physical', 'G_Decomp',
                           'G_Perceptual', 'G_Boundary',
                           'D_real', 'D_fake']

        self.visual_names = ['shadow', 'fake', 'real']
        if opt.isTrain and hasattr(opt, 'use_mask') and opt.use_mask:
            self.visual_names.append('mask')

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        self.netG = networks.UnfoldingGenerator(
            num_iterations=opt.num_iterations if hasattr(opt, 'num_iterations') else 3,
            ngf=opt.ngf,
            norm_layer=networks.get_norm_layer(opt.norm),
            use_dropout=not opt.no_dropout,
            log_eps=opt.log_eps if hasattr(opt, 'log_eps') else 1e-4,
            share_weights=opt.share_weights if hasattr(opt, 'share_weights') else False
        )

        if self.isTrain:

            self.fake_pool = ImagePool(opt.pool_size)

            self.netD = networks.define_D(
                opt.output_nc,
                opt.ndf, opt.netD,
                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain
            )

            # [修复] label smoothing: real label = 0.9
            label_smooth = opt.label_smoothing if hasattr(opt, 'label_smoothing') else 0.1
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode,
                target_real_label=1.0 - label_smooth,
                target_fake_label=0.0
            ).to(self.device)
            self.criterionPhysical = torch.nn.L1Loss()

            self.vgg_loss = VGGPerceptualLoss().to(self.device)

            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            # [修复] D学习率: lr * 0.1 (原来是 lr * 0.25)
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr * 0.1, betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # [修复] D更新频率计数器
            self.d_update_counter = 0
            self.d_update_freq = opt.d_update_freq if hasattr(opt, 'd_update_freq') else 2

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps."""
        if 'shadow' in input:
            self.shadow = input['shadow'].to(self.device)
            self.image_paths = input['shadow_paths']
        else:
            self.shadow = input['A'].to(self.device)
            self.image_paths = input['A_paths']

        if 'gt' in input:
            self.real = input['gt'].to(self.device)

        if 'mask' in input and input['mask'] is not None:
            self.mask = input['mask'].to(self.device)
        else:
            self.mask = torch.ones_like(self.shadow[:, :1])

        if 'shadow_paths' in input:
            self.image_paths = input['shadow_paths']

    def forward(self):
        if hasattr(self.opt, 'use_mask') and not self.opt.use_mask:
            mask_to_use = None
        else:
            mask_to_use = self.mask

        self.fake, self.intermediate_Js, self.last_J_log, self.A_log, self.S_log = self.netG(self.shadow, mask_to_use)

    def backward_D(self):
        # [修复] 使用 fake_pool
        fake_B = self.fake_pool.query(self.fake.detach())

        pred_fake = self.netD(fake_B)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # [修复] 给real加微弱噪声
        real_input = self.real + torch.randn_like(self.real) * 0.05
        real_input = real_input.clamp(-1.0, 1.0)

        pred_real = self.netD(real_input)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):

        pred_fake = self.netD(self.fake)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Physical Loss
        self.loss_G_Physical = 0
        mask_weight = (self.mask + 1.0) * 0.5
        mask_weight_factor = self.opt.mask_weight_factor if hasattr(self.opt, 'mask_weight_factor') else 2.0
        pixel_weight = 1.0 + mask_weight * mask_weight_factor

        num_iters = len(self.intermediate_Js)
        for i, fake_j in enumerate(self.intermediate_Js):
            abs_diff = torch.abs(fake_j - self.real)
            weighted_loss = torch.mean(abs_diff * pixel_weight)
            w = (i + 1) / num_iters
            self.loss_G_Physical += weighted_loss * w

        # Decomposition Consistency Loss
        diff_log = torch.abs(self.last_J_log + self.A_log - self.S_log)
        decomp_temp = self.opt.decomp_temp if hasattr(self.opt, 'decomp_temp') else 0.5
        valid_region = torch.sigmoid((self.S_log + 4.6) / decomp_temp).detach()
        self.loss_G_Decomp = torch.sum(diff_log * valid_region) / (torch.sum(valid_region) + 1e-8)

        # Perceptual Loss
        self.loss_G_Perceptual = self.vgg_loss(self.fake, self.real)

        # Boundary Loss
        self.loss_G_Boundary = self._boundary_loss(self.fake, self.real, self.mask)

        # Total Loss
        lambda_physical = self.opt.lambda_physical if hasattr(self.opt, 'lambda_physical') else 10.0
        lambda_gan = self.opt.lambda_gan if hasattr(self.opt, 'lambda_gan') else 1.5
        lambda_decomp = self.opt.lambda_decomp if hasattr(self.opt, 'lambda_decomp') else 3.0
        lambda_perceptual = self.opt.lambda_perceptual if hasattr(self.opt, 'lambda_perceptual') else 0.5
        lambda_boundary = self.opt.lambda_boundary if hasattr(self.opt, 'lambda_boundary') else 0.5

        self.loss_G = lambda_gan * self.loss_G_GAN + \
                      lambda_physical * self.loss_G_Physical + \
                      lambda_decomp * self.loss_G_Decomp + \
                      lambda_perceptual * self.loss_G_Perceptual + \
                      lambda_boundary * self.loss_G_Boundary

        self.loss_G.backward()

    def _boundary_loss(self, fake, real, mask):
        mask_01 = (mask + 1.0) * 0.5
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=fake.dtype, device=fake.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=fake.dtype, device=fake.device).view(1, 1, 3, 3)

        mask_grad_x = torch.nn.functional.conv2d(mask_01, sobel_x, padding=1)
        mask_grad_y = torch.nn.functional.conv2d(mask_01, sobel_y, padding=1)
        boundary_weight = (mask_grad_x ** 2 + mask_grad_y ** 2).sqrt()
        boundary_weight = boundary_weight / (boundary_weight.max() + 1e-8)

        loss = 0
        for c in range(fake.shape[1]):
            fake_c = fake[:, c:c+1, :, :]
            real_c = real[:, c:c+1, :, :]
            fake_gx = torch.nn.functional.conv2d(fake_c, sobel_x, padding=1)
            fake_gy = torch.nn.functional.conv2d(fake_c, sobel_y, padding=1)
            real_gx = torch.nn.functional.conv2d(real_c, sobel_x, padding=1)
            real_gy = torch.nn.functional.conv2d(real_c, sobel_y, padding=1)
            loss += torch.mean(boundary_weight * (torch.abs(fake_gx - real_gx) + torch.abs(fake_gy - real_gy)))

        return loss / fake.shape[1]

    # [修复] D每2步更新一次
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights"""
        self.forward()

        self.d_update_counter += 1
        if self.d_update_counter % self.d_update_freq == 0:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
        else:
            with torch.no_grad():
                fake_B = self.fake.detach()
                pred_fake = self.netD(fake_B)
                self.loss_D_fake = self.criterionGAN(pred_fake, False)
                pred_real = self.netD(self.real)
                self.loss_D_real = self.criterionGAN(pred_real, True)

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


class VGGPerceptualLoss(torch.nn.Module):

    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        except (ImportError, TypeError):
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True).features

        self.slice1 = torch.nn.Sequential(*list(vgg.children())[:4])   # relu1_2
        self.slice2 = torch.nn.Sequential(*list(vgg.children())[4:9])  # relu2_2
        self.slice3 = torch.nn.Sequential(*list(vgg.children())[9:16]) # relu3_3
        
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8]

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x):
        x = (x + 1.0) / 2.0
        return (x - self.mean) / self.std

    def forward(self, fake, real):
        fake_norm = self._normalize(fake)
        real_norm = self._normalize(real)

        loss = 0
        x_fake, x_real = fake_norm, real_norm
        for i, slice_layer in enumerate([self.slice1, self.slice2, self.slice3]):
            x_fake = slice_layer(x_fake)
            x_real = slice_layer(x_real)
            loss += self.weights[i] * torch.nn.functional.l1_loss(x_fake, x_real.detach())

        return loss
