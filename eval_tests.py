import torch
import argparse
import model
import utils
import time
from tqdm import tqdm
import numpy as np
import input_loaders
import random
if __name__ == '__main__':
    np.warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--csv')
    parser.add_argument('--icsv')
    parser.add_argument('--novid', action='store_true')
    parser.add_argument('--block_frames',type=int, default=0)
    parser.add_argument('--drop_frames',type=int, default=0)

    parser.add_argument('--embed')
    parser.add_argument('--mix', type=float)
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--audio_only', action='store_true')

    args = parser.parse_args()
    if args.novid:
        print('No video option is enabled')
    net = torch.nn.DataParallel(model.BigNet(301, utils.EMBEDDING_LENS[args.embed],audio_only=args.audio_only))
    print('opening file')

    if torch.cuda.is_available():
        weights = torch.load(args.model)
        net.cuda()
    else:
        weights = torch.load(args.model, map_location='cpu')

    net.load_state_dict(weights)
    net.eval()

    if args.noise:
        interference_root_dir = '/home/berkay/fastdata/Audioset/train'
        source_root_dir = utils.ROOT_DIRS[args.embed]
    else:
        interference_root_dir = utils.ROOT_DIRS[args.embed]
        source_root_dir = interference_root_dir
    eval_dataset = input_loaders.TrainDataset(args.csv, args.icsv,
                                              root_dir=source_root_dir,
                                              interference_root_dir=interference_root_dir,
                                              n_speakers=1, mix_coeff=args.mix, gradual=False)
    test_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=6,
                                                  shuffle=False, num_workers=6)
    output_SDR = 0
    output_pesq = 0
    output_CDIST = 0
    input_CDIST = 0
    input_SDR = 0
    input_pesq = 0
    output_SNR = 0
    input_SNR = 0

    input_STOI = 0
    output_STOI = 0

    with torch.no_grad():
        for i, sample in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Evaluation'):
            targets = sample['clean_spec']
            face_embs, mixed_spec = sample['face_embeds'], sample['mixed_spec']

            print(face_embs.shape)
            if args.novid:
                face_embs = torch.zeros_like(face_embs)
            elif args.block_frames > 0:
                blackout_start = random.randint(0,face_embs.shape[2]-args.block_frames)
                face_embs[:, :, blackout_start:blackout_start+args.block_frames] = 0

            elif args.drop_frames:
                face_embs[:, :, ::args.drop_frames] = 0

            if torch.cuda.is_available():
                targets, face_embs, mixed_spec = targets.cuda(), face_embs.cuda(), mixed_spec.cuda()
            outputs = net(face_embs, mixed_spec)

            output_SDR += utils.calculate_SDR(outputs, targets)
            input_SDR += utils.calculate_SDR(mixed_spec, targets)

            if not args.fast:
                output_pesq += utils.get_pesq(outputs, targets)
                print('input ones')
                input_pesq += utils.get_pesq(mixed_spec, targets)
                # output_CDIST += utils.get_cep_distance(outputs, targets)
                # input_CDIST += utils.get_cep_distance(mixed_spec, targets)


            input_SNR += utils.get_SNR(targets, mixed_spec)
            output_SNR += utils.get_SNR(targets, outputs)

            input_STOI += utils.get_STOI(mixed_spec,targets)
            output_STOI += utils.get_STOI(outputs,targets)

        output_pesq /= len(test_dataloader.dataset)
        input_pesq /= len(test_dataloader.dataset)

        output_SDR /= len(test_dataloader.dataset)
        input_SDR /= len(test_dataloader.dataset)

        output_CDIST /= len(test_dataloader.dataset)
        input_CDIST /= len(test_dataloader.dataset)


        input_SNR /= len(test_dataloader.dataset)
        output_SNR /= len(test_dataloader.dataset)

        input_STOI /= len(test_dataloader.dataset)
        output_STOI /= len(test_dataloader.dataset)

    utils.SingletonPool.close_pool()

    # print('OUTPUT PESQ:', output_pesq)
    # print('INPUT PESQ:', input_pesq)
    #
    # print('INPUT SDR:', input_SDR, 'dB')
    # print('OUTPUT SDR:', output_SDR, 'dB')
    #
    # print('INPUT SNR:', input_SNR, 'dB')
    # print('OUTPUT SNR:', output_SNR, 'dB')
    #
    # print('CDIST:', cdist_db, 'dB')


    with open('/tmp/eval_log_{}.txt'.format(time.strftime("%Y%m%d_%H%M%S")), 'w') as f:
        f.write('INPUT PESQ: {}\n'.format(input_pesq))
        f.write('OUTPUT PESQ: {}\n'.format(output_pesq))

        f.write('INPUT SDR: {} dB\n'.format(input_SDR))
        f.write('OUTPUT SDR: {} dB\n'.format(output_SDR))

        f.write('INPUT SNR: {} dB\n'.format(input_SNR))
        f.write('OUTPUT SNR: {} dB\n'.format(output_SNR))

        f.write('INPUT STOI: {} \n'.format(input_STOI))
        f.write('OUTPUT STOI: {}\n'.format(output_STOI))

        f.write('INPUT CDIST: {} dB\n'.format(input_CDIST))
        f.write('OUTPUT CDIST: {} dB\n'.format(output_CDIST))


        f.write('\n{}\n'.format(str(args)))


    with open('/tmp/eval_log_{}.txt'.format(time.strftime("%Y%m%d_%H%M%S"))) as f:
        for line in f:
            print(line)

