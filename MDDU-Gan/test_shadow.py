#!/usr/bin/env python3
"""
Simple testing script for shadow removal model.

Example usage:
    # Test on dataset with ground truth
    python test_shadow.py --dataroot ./datasets/ISTD --name shadow_removal_istd --phase test
    
    # Test on single images (no ground truth needed)
    python test_shadow.py --dataroot ./test_images/shadow --name shadow_removal_istd \
        --dataset_mode single --phase test
"""

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import os
import torch
if __name__ == '__main__':
    # Parse options
    opt = TestOptions().parse()
    opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not hasattr(opt, 'gpu_ids'):
        opt.gpu_ids = [0] if torch.cuda.is_available() else []
    # Hard-code some parameters for shadow removal testing
    opt.num_threads = 0   # shadow removal testing only supports num_threads = 0
    opt.batch_size = 1    # shadow removal testing only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    
    # Override model and dataset mode for shadow removal (if not using single dataset mode)
    if opt.dataset_mode != 'single':
        opt.model = 'shadow_removal'
        opt.dataset_mode = 'shadow'
    else:
        opt.model = 'shadow_removal'
    
    # Create dataset
    dataset = create_dataset(opt)
    
    # Create model
    model = create_model(opt)
    model.setup(opt)
    
    # Create a website
    web_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
    if opt.load_iter > 0:
        web_dir = f'{web_dir}_{opt.load_iter}'
    print(f'creating web directory {web_dir}')
    webpage = html.HTML(web_dir, f'Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}')
    
    # Test with eval mode
    if opt.eval:
        model.eval()
    
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply to a subset of the data
            break
        
        model.set_input(data)
        model.test()
        
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        
        if i % 5 == 0:
            print(f'processing ({i:04d})-th image... {img_path}')
        
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    
    webpage.save()
    print(f'\nResults saved to: {web_dir}')
