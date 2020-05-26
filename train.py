import torch
from utils.dataset import CustomerDataset, CustomerCollate
from torch.utils.data import DataLoader
import torch.nn.parallel.data_parallel as parallel
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import time
import tqdm
from models.generator import Generator
from models.multiscale import MultiScaleDiscriminator
from utils.writer import Writer
from utils.optimizer import Optimizer
from utils.audio import hop_length, sample_rate
from utils.logging import GetLogging
import torch.nn.functional as F
from utils.loss import MultiResolutionSTFTLoss

def create_model(args):

    generator = Generator(args.local_condition_dim)
    discriminator = MultiScaleDiscriminator() 
 
    return generator, discriminator

def save_checkpoint(args, generator, discriminator, g_optimizer,
                    d_optimizer, step, logging):
    checkpoint_path = os.path.join(args.checkpoint_dir, "model.ckpt-{}.pt".format(step))

    torch.save({"generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "g_optimizer": g_optimizer.state_dict(),
                "d_optimizer": d_optimizer.state_dict(),
                "global_step": step
                }, checkpoint_path)

    logging.info("Saved checkpoint: {}".format(checkpoint_path))

    with open(os.path.join(args.checkpoint_dir, 'checkpoint'), 'w') as f:
        f.write("model.ckpt-{}.pt".format(step))

def attempt_to_restore(generator, discriminator, g_optimizer, d_optimizer,
                       checkpoint_dir, use_cuda, logging):
    checkpoint_list = os.path.join(checkpoint_dir, 'checkpoint')

    if os.path.exists(checkpoint_list):
        checkpoint_filename = open(checkpoint_list).readline().strip()
        checkpoint_path = os.path.join(
            checkpoint_dir, "{}".format(checkpoint_filename))
        logging.info("Restore from {}".format(checkpoint_path))
        checkpoint = load_checkpoint(checkpoint_path, use_cuda)
        generator.load_state_dict(checkpoint["generator"])
        g_optimizer.load_state_dict(checkpoint["g_optimizer"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        d_optimizer.load_state_dict(checkpoint["d_optimizer"])
        global_step = checkpoint["global_step"]

    else:
        global_step = 0

    return global_step

def load_checkpoint(checkpoint_path, use_cuda):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)

    return checkpoint

def train(args):

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logging = GetLogging(args.logfile)

    train_dataset = CustomerDataset(
         args.input,
         upsample_factor=hop_length,
         local_condition=True,
         global_condition=False)

    device = torch.device("cuda" if args.use_cuda else "cpu")
    generator, discriminator = create_model(args)

    print(generator)
    print(discriminator)

    num_gpu = torch.cuda.device_count() if args.use_cuda else 1

    global_step = 0

    g_parameters = list(generator.parameters())
    g_optimizer = optim.Adam(g_parameters, lr=args.g_learning_rate)

    d_parameters = list(discriminator.parameters())
    d_optimizer = optim.Adam(d_parameters, lr=args.d_learning_rate)
    
    writer = Writer(args.checkpoint_dir, sample_rate=sample_rate)

    generator.to(device)
    discriminator.to(device)

    if args.resume is not None:
        restore_step = attempt_to_restore(generator, discriminator, g_optimizer,
                               d_optimizer, args.resume, args.use_cuda, logging)
        global_step = restore_step

    global_step = 100000
    customer_g_optimizer = Optimizer(g_optimizer, args.g_learning_rate,
                global_step, args.warmup_steps, args.decay_learning_rate)
    customer_d_optimizer = Optimizer(d_optimizer, args.d_learning_rate,
                global_step, args.warmup_steps, args.decay_learning_rate)

    criterion = nn.MSELoss().to(device)
    stft_criterion = MultiResolutionSTFTLoss()

    for epoch in range(args.epochs):

       collate = CustomerCollate(
           upsample_factor=hop_length,
           condition_window=args.condition_window,
           local_condition=True,
           global_condition=False)

       train_data_loader = DataLoader(train_dataset, collate_fn=collate,
               batch_size=args.batch_size, num_workers=args.num_workers,
               shuffle=True, pin_memory=True)

       #train one epoch
       for batch, (samples, conditions) in enumerate(train_data_loader):

            start = time.time()
            batch_size = int(conditions.shape[0] // num_gpu * num_gpu)

            samples = samples[:batch_size, :].to(device)
            conditions = conditions[:batch_size, :, :].to(device)

            losses = {}

            if num_gpu > 1:
                g_outputs = parallel(generator, (conditions, ))
            else:
                g_outputs = generator(conditions)

            #train discriminator
            if global_step > args.discriminator_train_start_steps:
                if num_gpu > 1:
                    fake_scores = parallel(discriminator, (g_outputs.detach(),))
                    real_scores = parallel(discriminator, (samples,))

                else:
                    fake_scores = discriminator(g_outputs.detach())
                    real_scores = discriminator(samples)

                d_loss_fake_list, d_loss_real_list = [], []
                for (real_score, fake_score) in zip(real_scores, fake_scores):
                    d_loss_real_list.append(criterion(real_score, torch.ones_like(real_score)))
                    d_loss_fake_list.append(criterion(fake_score, torch.zeros_like(fake_score)))

                d_loss_real = sum(d_loss_real_list) / len(d_loss_real_list)
                d_loss_fake = sum(d_loss_fake_list) / len(d_loss_fake_list)
                d_loss = d_loss_real + d_loss_fake

                for (i, _) in enumerate(d_loss_real_list):
                    losses['d_loss_real_{}'.format(str(i))] = d_loss_real_list[i].item()
                    losses['d_loss_fake_{}'.format(str(i))] = d_loss_fake_list[i].item()

                customer_d_optimizer.zero_grad()
                d_loss.backward()
                nn.utils.clip_grad_norm_(d_parameters, max_norm=0.5)
                customer_d_optimizer.step_and_update_lr()
            else:
                d_loss_real = torch.Tensor([0])
                d_loss_fake = torch.Tensor([0])
                d_loss = torch.Tensor([0])

            losses['d_loss_real'] = d_loss_real.item()
            losses['d_loss_fake'] = d_loss_fake.item()
            losses['d_loss'] = d_loss.item()

            #train generator
            if global_step > args.discriminator_train_start_steps:
                if num_gpu > 1:
                    fake_scores = parallel(discriminator, (g_outputs,))
                else:
                    fake_scores = discriminator(g_outputs)

                adv_loss_list = []
                for fake_score in fake_scores:
                    adv_loss_list.append(criterion(fake_score, torch.ones_like(fake_score)))
                adv_loss = sum(adv_loss_list) / len(adv_loss_list)

                for (i, loss) in enumerate(adv_loss_list):
                    losses["adv_loss_{}".format(str(i))] = loss.item()
            else:
                adv_loss = torch.Tensor([0])

            sc_loss, mag_loss = stft_criterion(g_outputs.squeeze(1), samples.squeeze(1))

            if global_step > args.discriminator_train_start_steps:
                g_loss = adv_loss * args.lamda_adv + sc_loss + mag_loss
            else:
                g_loss = sc_loss + mag_loss
           
            losses['adv_loss'] = adv_loss.item()
            losses['sc_loss'] = sc_loss.item()
            losses['mag_loss'] = mag_loss.item()
            losses['g_loss'] = g_loss.item()

            customer_g_optimizer.zero_grad()
            g_loss.backward()
            nn.utils.clip_grad_norm_(g_parameters, max_norm=0.5)
            customer_g_optimizer.step_and_update_lr()

            time_used = time.time() - start

            if global_step > args.discriminator_train_start_steps:
                logging.info("Step: {} --adv_loss: {:.3f} --sc_loss: {:.3f} --mag_loss: {:.3f} --real_loss: {:.3f} --fake_loss: {:.3f} --Time: {:.2f} seconds".format(global_step, adv_loss, sc_loss, mag_loss, d_loss_real, d_loss_fake, time_used))
            else:
                logging.info("Step: {} --sc_loss: {:.3f} --mag_loss: {:.3f} --Time: {:.2f} seconds".format(
                   global_step, sc_loss, mag_loss, time_used))

            if global_step % args.checkpoint_step ==0:
                save_checkpoint(args, generator, discriminator,
                     g_optimizer, d_optimizer, global_step, logging)
                
            if global_step % args.summary_step == 0:
                writer.logging_loss(losses, global_step)
                target = samples.cpu().detach()[0, 0].numpy()
                predict = g_outputs.cpu().detach()[0, 0].numpy()
                writer.logging_audio(target, predict, global_step)
                writer.logging_histogram(generator, global_step)
                writer.logging_histogram(discriminator, global_step)

            global_step += 1

def main():

    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/train', help='Directory of training data')
    parser.add_argument('--num_workers',type=int, default=1, help='Number of dataloader workers.')
    parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--checkpoint_dir', type=str, default="logdir", help="Directory to save model")
    parser.add_argument('--resume', type=str, default=None, help="The model name to restore")
    parser.add_argument('--checkpoint_step', type=int, default=5000)
    parser.add_argument('--summary_step', type=int, default=100)
    parser.add_argument('--use_cuda', type=_str_to_bool, default=True)
    parser.add_argument('--g_learning_rate', type=float, default=0.0001)
    parser.add_argument('--d_learning_rate', type=float, default=0.0001)
    parser.add_argument('--warmup_steps', type=int, default=100000)
    parser.add_argument('--decay_learning_rate', type=float, default=0.5)
    parser.add_argument('--local_condition_dim', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--condition_window', type=int, default=100)
    parser.add_argument('--lamda_adv', type=float, default=2.5)
    parser.add_argument('--logfile', type=str, default="txt")
    parser.add_argument('--discriminator_train_start_steps', type=int, default=100000)

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
