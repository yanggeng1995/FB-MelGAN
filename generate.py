import torch
from utils.audio import save_wav
import argparse
import os
import time
import numpy as np
from models.generator import Generator

def attempt_to_restore(generate, checkpoint_dir, use_cuda):
    checkpoint_list = os.path.join(checkpoint_dir, 'checkpoint')

    if os.path.exists(checkpoint_list):
        checkpoint_filename = open(checkpoint_list).readline().strip()
        checkpoint_path = os.path.join(
            checkpoint_dir, "{}".format(checkpoint_filename))
        print("Restore from {}".format(checkpoint_path))
        checkpoint = load_checkpoint(checkpoint_path, use_cuda)
        generate.load_state_dict(checkpoint["generator"])

def load_checkpoint(checkpoint_path, use_cuda):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                map_location=lambda storage, loc: storage)

    return checkpoint

def create_model(args):

    generator = Generator(args.local_condition_dim)
    
    return generator 

def synthesis(args):

    model = create_model(args)
    model.eval()

    if args.resume is not None:
       attempt_to_restore(model, args.resume, args.use_cuda)

    device = torch.device("cuda" if args.use_cuda else "cpu")
    model.to(device)
    model.remove_weight_norm()

    output_dir = "samples"
    target_dir = os.path.join(output_dir, "target")
    predict_dir = os.path.join(output_dir, "predict")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(predict_dir, exist_ok=True)

    avg_rtf = []
    for filename in os.listdir(os.path.join(args.input, 'mel')):
        start = time.time()
        conditions = np.load(os.path.join(args.input, 'mel', filename))
        conditions = torch.FloatTensor(conditions).unsqueeze(0)
        conditions = conditions.transpose(1, 2).to(device)
        audio = model(conditions)
        audio = audio.cpu().squeeze().detach().numpy()
        print(audio.shape)
        name = filename.split('.')[0]
        sample = np.load(os.path.join(args.input, 'audio', filename))
        save_wav(np.squeeze(sample), '{}/{}_target.wav'.format(target_dir, name))
        save_wav(np.asarray(audio), '{}/{}.wav'.format(predict_dir, name))
        time_used = time.time() - start
        rtf = time_used / (len(audio) / 16000)
        avg_rtf.append(rtf)
        print("Time used: {:.3f}, RTF: {:.4f}".format(time_used, rtf))
    print("Average RTF: {:.3f}".format(sum(avg_rtf) / len(avg_rtf)))

def main():

    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]


    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='test', help='Directory of tests data')
    parser.add_argument('--num_workers',type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument('--resume', type=str, default="logdir")
    parser.add_argument('--local_condition_dim', type=int, default=80)
    parser.add_argument('--use_cuda', type=_str_to_bool, default=False)

    args = parser.parse_args()
    synthesis(args)

if __name__ == "__main__":
    main()
