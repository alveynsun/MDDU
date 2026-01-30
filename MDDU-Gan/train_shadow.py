#!/usr/bin/env python3
"""
Simple training script for shadow removal model.

Example usage:
    python train_shadow.py --dataroot ./datasets/ISTD --name shadow_removal_istd

    python train_shadow.py --dataroot ./datasets/ISTD --name shadow_removal_istd \
        --num_iterations 3 --lambda_physical 10.0 --use_mask
"""

import sys
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp
import time

if __name__ == '__main__':
    # Parse options
    opt = TrainOptions().parse()

    # Override model and dataset mode for shadow removal
    opt.model = 'shadow_removal'
    opt.dataset_mode = 'shadow'

    # Initialize DDP if needed
    opt.device = init_ddp()

    # Create dataset
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f'The number of training images = {dataset_size}')

    # Create model
    model = create_model(opt)
    model.setup(opt)

    # Create visualizer
    visualizer = Visualizer(opt)

    # Training loop
    total_iters = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        # Set epoch for DistributedSampler
        if hasattr(dataset, 'set_epoch'):
            dataset.set_epoch(epoch)

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            # Train
            model.set_input(data)
            model.optimize_parameters()

            # Display results
            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            # Print losses
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if hasattr(opt, 'display_id') and opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            # Save latest model
            if total_iters % opt.save_latest_freq == 0:
                print(f'saving the latest model (epoch {epoch}, total_iters {total_iters})')
                save_suffix = f'iter_{total_iters}' if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # Save model at end of epoch
        if epoch % opt.save_epoch_freq == 0:
            print(f'saving the model at the end of epoch {epoch}, iters {total_iters}')
            model.save_networks('latest')
            model.save_networks(epoch)

        # Update learning rates
        print(
            f'End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time:.2f} sec')
        model.update_learning_rate()

    # Cleanup
    cleanup_ddp()