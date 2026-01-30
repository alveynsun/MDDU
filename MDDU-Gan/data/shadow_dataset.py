import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class ShadowDataset(BaseDataset):
    """A dataset class for shadow removal.

    Expected directory structure:
    dataroot/
        train/
            shadow/    # Shadow images
            gt/        # Ground truth (shadow-free)
            mask/      # Shadow masks (optional)
        test/
            shadow/
            gt/        # Optional for test
            mask/      # Optional

    Returns a dictionary with:
        'shadow': shadow image tensor
        'gt': ground truth tensor (training only)
        'mask': shadow mask tensor (can be None or all-ones if not available)
        'shadow_paths': image path
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser -- original option parser
            is_train (bool) -- whether training phase or test phase

        Returns:
            the modified parser.
        """
        parser.add_argument('--mask_dropout_prob', type=float, default=0.0,
                            help='probability of replacing mask with all-ones during training')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags
        """
        BaseDataset.__init__(self, opt)

        # Get directory paths
        self.dir_shadow = os.path.join(opt.dataroot, opt.phase, 'shadow')
        self.dir_gt = os.path.join(opt.dataroot, opt.phase, 'gt')
        self.dir_mask = os.path.join(opt.dataroot, opt.phase, 'mask')

        # Get file paths
        self.shadow_paths = sorted(make_dataset(self.dir_shadow, opt.max_dataset_size))

        # Check if ground truth exists
        self.has_gt = os.path.isdir(self.dir_gt)
        if self.has_gt:
            self.gt_paths = sorted(make_dataset(self.dir_gt, opt.max_dataset_size))
            # Verify that shadow and gt have same number of images
            assert len(self.shadow_paths) == len(self.gt_paths), \
                f"Shadow and GT directories have different number of images: " \
                f"{len(self.shadow_paths)} vs {len(self.gt_paths)}"

        # Check if masks exist
        self.has_mask = os.path.isdir(self.dir_mask)
        if self.has_mask:
            self.mask_paths = sorted(make_dataset(self.dir_mask, opt.max_dataset_size))

        self.is_train = opt.isTrain

        # Mask dropout probability for training
        self.mask_dropout_prob = opt.mask_dropout_prob if hasattr(opt, 'mask_dropout_prob') else 0.0

        # Verify we have the correct setup
        assert len(self.shadow_paths) > 0, f"No images found in {self.dir_shadow}"

        print(f"Shadow dataset initialized:")
        print(f"  - Shadow images: {len(self.shadow_paths)}")
        print(f"  - Has ground truth: {self.has_gt}")
        print(f"  - Has masks: {self.has_mask}")

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            A dictionary containing:
                'shadow': shadow image tensor
                'gt': ground truth tensor (if available)
                'mask': shadow mask tensor
                'shadow_paths': image path
        """
        # Get paths
        shadow_path = self.shadow_paths[index]

        # Load shadow image
        shadow = Image.open(shadow_path).convert('RGB')

        # Load ground truth if available
        if self.has_gt:
            gt_path = self.gt_paths[index]
            gt = Image.open(gt_path).convert('RGB')

        # Load or create mask
        mask_available = False
        if self.has_mask and index < len(self.mask_paths):
            mask_path = self.mask_paths[index]
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')  # Grayscale
                mask_available = True

        if not mask_available:
            # Create all-ones mask (no mask information)
            mask = Image.new('L', shadow.size, 255)

        # Apply mask dropout during training
        if self.is_train and random.random() < self.mask_dropout_prob:
            mask = Image.new('L', shadow.size, 255)

        # Apply transforms
        # Use the same transform parameters for shadow, gt, and mask to ensure alignment
        transform_params = get_params(self.opt, shadow.size)

        shadow_transform = get_transform(self.opt, transform_params, grayscale=False)
        mask_transform = get_transform(self.opt, transform_params, grayscale=True)

        shadow = shadow_transform(shadow)
        mask = mask_transform(mask)

        result = {
            'shadow': shadow,
            'mask': mask,
            'shadow_paths': shadow_path
        }

        if self.has_gt:
            gt_transform = get_transform(self.opt, transform_params, grayscale=False)
            gt = gt_transform(gt)
            result['gt'] = gt

        return result

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.shadow_paths)