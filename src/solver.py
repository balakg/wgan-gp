import numpy as np
import os
import time
import datetime

import torch
from torchvision.utils import save_image

import losses
from data_loader import get_loader
from model import Generator, Discriminator


class Solver(object):
    """Solver for training and testing."""

    def __init__(self, config):
        # Data loaders.
        self.data_loader = get_loader(config.batch_size, config.im_size, config.train)

        # Model configurations.
        self.lambda_gp = config.lambda_gp
        self.z_dim = config.z_dim
        self.im_size = config.im_size

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.D_loss = losses.D_wgan_gp
        self.G_loss = losses.G_wgan

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_dir = config.model_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        self.build_model()


    def build_model(self):
        """Create a generator and a discriminator."""
        beta1, beta2 = 0.5, 0.999
       
        self.G = Generator(self.z_dim, self.im_size)
        self.G.to(self.device)
        self.G_opt = torch.optim.Adam(self.G.parameters(), self.g_lr, [beta1, beta2])
        #self.print_network(self.G, 'G')

        self.D = Discriminator(self.im_size)
        self.D.to(self.device)
        self.D_opt = torch.optim.Adam(self.D.parameters(), self.d_lr, [beta1, beta2])
        #self.print_network(self.D, 'D')


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))


    def reset_grad(self):
        """Reset the gradient buffers."""
        self.G_opt.zero_grad()
        self.D_opt.zero_grad()


    def train(self):
        print('Start training...')
        start_time = time.time()
        for i in range(0, self.num_iters):

            # Logging.
            loss = {}

            # 1. Train the discriminator
            try:
                x = next(iterator)
            except:
                iterator = iter(self.data_loader)
                x = next(iterator)

            x = x.to(self.device)

            d_loss = self.D_loss(self.G, self.D, x, self.z_dim, self.batch_size, self.lambda_gp, self.device)
            self.reset_grad()
            d_loss.backward()
            self.D_opt.step()

            loss['D/loss'] = d_loss.item()

            #2. Train the generator every n_critic iterations.
            if (i+1) % self.n_critic == 0:
                g_loss = self.G_loss(self.G, self.D, self.z_dim, self.batch_size, self.device)

                self.reset_grad()
                g_loss.backward()
                self.G_opt.step()

                loss['G/loss'] = g_loss.item()

  
            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)


            # Sample images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    z = torch.rand((self.batch_size, self.z_dim)).to(self.device)
                    x = self.G(z)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(x.data.cpu(), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))


            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_dir, '{}-G.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                print('Saved model checkpoints into {}...'.format(self.model_dir))


    def test(self):
        pass
