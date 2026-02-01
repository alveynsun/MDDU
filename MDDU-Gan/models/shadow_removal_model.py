import torch
from .base_model import BaseModel
from . import networks


class ShadowRemovalModel(BaseModel):
    """This class implements the shadow removal model using deep unfolding.

    The model is based on the Retinex theory in logarithmic space:
        log(S) = log(J) + log(A)
    where S is the shadow image, J is the reflectance (shadow-free), and A is illumination.

    It uses an iterative unfolding process with K stages to progressively refine
    the shadow-free output, guided by adaptive parameters from a HyperNet.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser -- original option parser
            is_train (bool) -- whether training phase or test phase

        Returns:
            the modified parser.
        """
        # Set default values for shadow removal
        # Note: netG is set to 'unet_256' for compatibility but UnfoldingGenerator is actually used
        parser.set_defaults(norm='batch', dataset_mode='shadow')
        parser.set_defaults(input_nc=3, output_nc=3)

        if is_train:
            parser.set_defaults(pool_size=50, gan_mode='lsgan')
            parser.add_argument('--num_iterations', type=int, default=3,
                                help='K iterations in unfolding process')
            parser.add_argument('--use_mask', action='store_true',
                                help='use shadow mask if available')
            parser.add_argument('--lambda_physical', type=float, default=5.0,
                                help='weight for physical reconstruction loss')
            parser.add_argument('--lambda_gan', type=float, default=5.0,
                                help='weight for adversarial loss')
            parser.add_argument('--lambda_decomp', type=float, default=0.3,
                                help='weight for decomposition loss')
            parser.add_argument('--mask_weight_factor', type=float, default=2.0,
                                help='extra weight for shadow regions in physical loss')
            parser.add_argument('--log_eps', type=float, default=1e-4,
                                help='epsilon for log transform to avoid log(0)')
            parser.add_argument('--decomp_temp', type=float, default=0.5,
                                help='temperature for soft valid region in decomposition loss')
            parser.add_argument('--share_weights', action='store_true',
                                help='share weights across unfolding iterations')

        return parser

    def __init__(self, opt):
        """Initialize the shadow removal model.

        Parameters:
            opt (Option class) -- stores all the experiment flags
        """
        BaseModel.__init__(self, opt)

        # Specify the training losses you want to print out
        self.loss_names = ['G_GAN', 'G_Physical', 'G_Decomp', 'D_real', 'D_fake']

        # Specify the images you want to save/display
        # 'shadow' is input, 'fake' is generated shadow-free, 'real' is ground truth
        self.visual_names = ['shadow', 'fake', 'real']
        if opt.isTrain and hasattr(opt, 'use_mask') and opt.use_mask:
            self.visual_names.append('mask')

        # Specify the models you want to save to the disk
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        # Define the generator (UnfoldingGenerator)
        self.netG = networks.UnfoldingGenerator(
            num_iterations=opt.num_iterations if hasattr(opt, 'num_iterations') else 3,
            ngf=opt.ngf,
            norm_layer=networks.get_norm_layer(opt.norm),
            use_dropout=not opt.no_dropout,
            log_eps=opt.log_eps if hasattr(opt, 'log_eps') else 1e-4,
            share_weights=opt.share_weights if hasattr(opt, 'share_weights') else False
        )

        if self.isTrain:
            # Define the discriminator
            # Discriminator takes concatenated (shadow, shadow-free) pairs
            self.netD = networks.define_D(
                opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain
            )

            # Define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionPhysical = torch.nn.L1Loss()

            # Initialize optimizers
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr * 0.5, betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        Expected keys:
            'shadow': shadow image tensor
            'gt': ground truth shadow-free image tensor (training only)
            'mask': shadow mask tensor (optional)
            'shadow_paths': image file paths
        """
        if 'shadow' in input:
            self.shadow = input['shadow'].to(self.device)
            self.image_paths = input['shadow_paths']
        else:
            self.shadow = input['A'].to(self.device)
            self.image_paths = input['A_paths']

        if 'gt' in input:
            self.real = input['gt'].to(self.device)

        # Handle optional mask
        if 'mask' in input and input['mask'] is not None:
            self.mask = input['mask'].to(self.device)
        else:
            # Create all-ones mask if not provided
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
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake
        fake_pair = torch.cat([self.shadow, self.fake.detach()], dim=1)
        pred_fake = self.netD(fake_pair)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_pair = torch.cat([self.shadow, self.real], dim=1)
        pred_real = self.netD(real_pair)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and physical losses for the generator"""

        fake_pair = torch.cat([self.shadow, self.fake], dim=1)
        pred_fake = self.netD(fake_pair)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_Physical = 0

        mask_weight = (self.mask + 1.0) * 0.5
        mask_weight_factor = self.opt.mask_weight_factor if hasattr(self.opt, 'mask_weight_factor') else 2.0
        pixel_weight = 1.0 + mask_weight * mask_weight_factor

        # 遍历所有中间结果
        for fake_j in self.intermediate_Js:

            abs_diff = torch.abs(fake_j - self.real)
            # 广播权重并求平均
            weighted_loss = torch.mean(abs_diff * pixel_weight)
            self.loss_G_Physical += weighted_loss

        self.loss_G_Physical = self.loss_G_Physical / len(self.intermediate_Js)
        # ================== 修改结束 ==================

        # 3. Decomp Loss
        diff_log = torch.abs(self.last_J_log + self.A_log - self.S_log)


        decomp_temp = self.opt.decomp_temp if hasattr(self.opt, 'decomp_temp') else 0.5
        valid_region = torch.sigmoid((self.S_log + 4.6) / decomp_temp).detach()

        self.loss_G_Decomp = torch.sum(diff_log * valid_region) / (torch.sum(valid_region) + 1e-8)

        # 4. 组合 Loss
        lambda_physical = self.opt.lambda_physical if hasattr(self.opt, 'lambda_physical') else 10.0
        lambda_gan = self.opt.lambda_gan if hasattr(self.opt, 'lambda_gan') else 1.0
        lambda_decomp = self.opt.lambda_decomp if hasattr(self.opt, 'lambda_decomp') else 1.0

        self.loss_G = lambda_gan * self.loss_G_GAN + \
                      lambda_physical * self.loss_G_Physical + \
                      lambda_decomp * self.loss_G_Decomp

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights"""
        self.forward()  # Compute fake images: G(shadow)

        # Update D
        self.set_requires_grad(self.netD, True)  # Enable backprop for D
        self.optimizer_D.zero_grad()  # Set D's gradients to zero
        self.backward_D()  # Calculate gradients for D
        self.optimizer_D.step()  # Update D's weights

        # Update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # Set G's gradients to zero
        self.backward_G()  # Calculate gradients for G
        self.optimizer_G.step()  # Update G's weights
