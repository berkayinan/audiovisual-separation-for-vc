import model
import argparse
import torch
import numpy as np
import librosa
from tqdm import tqdm
import utils
import os
import input_loaders
import time
import datetime
import mir_eval
import tensorboardX


def train(args):
    print(args)

    assert args.n_speaker >= 1
    print('N_SPEAKERS',args.n_speaker)
    MULTI_GPU = True

    EMBEDDING_LEN = 1792
    for embed_type in utils.EMBEDDING_LENS:
        if embed_type in args.root_dir:
            EMBEDDING_LEN = utils.EMBEDDING_LENS[embed_type]

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print('CUDA_VISIBLE_DEVICES is set to {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    else:
        print('No restriction of CUDA Devices')
    if args.n_speaker == 1:
        net = model.BigNet(n_time_bins=301, embedding_len=EMBEDDING_LEN, audio_only=args.audio_only)
    else:
        net = model.DoubleSpeakerNet(n_time_bins=301,n_speakers=2,embedding_len=EMBEDDING_LEN,
                                     audio_only=args.audio_only)
    if MULTI_GPU:
        print('MULTI GPU')
        net = torch.nn.DataParallel(net)
    net = net.cuda()



    loss_criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr)

    logger = utils.MyLogger(suffix=args.suffix)
    logger.write_to_log(str(args), 'train_args.txt')

    # train_dataset = input_loaders.SingleLoader()
    # test_dataset = input_loaders.SingleLoader()

    train_dataset = input_loaders.TrainDataset(input_table=args.train, root_dir=args.root_dir,
                                               interference_table=args.interference_train, interference_root_dir=args.interference_root_dir,
                                               n_speakers=args.n_speaker,
                                               mix_coeff=args.mix,gradual =args.gradual)
    test_dataset = input_loaders.TrainDataset(input_table=args.eval,root_dir=args.root_dir,
                                              interference_table=args.interference_eval, interference_root_dir=args.interference_root_dir,
                                              n_speakers=args.n_speaker,
                                              mix_coeff=args.mix,gradual =args.gradual,
                                              testing=True)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=6,
                                                   shuffle=True, num_workers=6)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=6,
                                                  shuffle=False, num_workers=6)


    os.makedirs('{}/train_checkpoints'.format(logger.logs_dir),exist_ok=True)
    with tensorboardX.SummaryWriter() as tb_writer :
        for epoch in range(100000000):

            # for name, param in net.named_parameters():
            #     tb_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            print('EPOCH', epoch)
            total_loss = 0
            if epoch > 0 and epoch%10 == 0 and args.gradual:
                train_dataset.update_mix_factor()
                test_dataset.update_mix_factor()

            if epoch % 1 == 0:
                torch.save(net.state_dict(),
                           '{}/models/model_epoch_{}_loss_{}.pth'.format(logger.logs_dir, epoch, total_loss))
                torch.save({'epoch':epoch,'optimizer':optimizer,'logs':logger},
                           '{}/train_checkpoints/checkpoint_{}.pth'.format(logger.logs_dir, epoch))
                evaluate(test_dataloader, net, loss_criterion, epoch, logger,args)


            for i, sample_batched in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Training'):
                outputs, loss = train_step(net, optimizer, loss_criterion, sample_batched, args)
                # utils.reconstruct_wav_from_spec(outputs[i],'{}/trainout_spec_example_epoch_{}_ex_{}.wav'.format(logger.logs_dir,epoch,i))

                total_loss += loss.item()
                if i > 0 and i % 100 == 0:
                    print('[', epoch, i, ']', total_loss / i)
            total_loss /= len(train_dataloader.dataset)

            logger.write_to_log('{}, {}'.format(epoch, total_loss), 'train_loss.csv')
            logger.write_to_log('{}, {}'.format(epoch, train_dataset.mix_factor), 'mix_factor.csv')



def evaluate(loader, net, loss_criterion, epoch, logger,args):
    net.eval()
    with torch.no_grad():
        single_sample = next(iter(loader))
        targets = single_sample['clean_spec'].cuda()
        face_embs, mixed_spec = single_sample['face_embeds'].cuda(), single_sample['mixed_spec'].cuda()
        outputs = net(face_embs, mixed_spec)
        example_dir = os.path.join(logger.logs_dir, 'examples')
        os.makedirs(example_dir, exist_ok=True)
        if args.n_speaker == 1:
            for i in range(len(outputs)):
                utils.reconstruct_wav_from_spec(targets[i],
                                                '{}/clean_spec_example_epoch_{}_ex_{}.wav'.format(example_dir, epoch, i))
                utils.reconstruct_wav_from_spec(outputs[i],
                                                '{}/output_spec_example_epoch_{}_ex_{}.wav'.format(example_dir, epoch, i))
                utils.reconstruct_wav_from_spec(mixed_spec[i],
                                                '{}/mixed_spec_example_epoch_{}_ex_{}.wav'.format(example_dir, epoch, i))
        else:
            for i in range(len(outputs)):
                for s in range(args.n_speaker):
                    utils.reconstruct_wav_from_spec(targets[i,s],
                                                    '{}/clean_spec_example_epoch_{}_ex_{}_speaker_{}.wav'.format(example_dir, epoch, i,s))
                    utils.reconstruct_wav_from_spec(outputs[i,s],
                                                    '{}/output_spec_example_epoch_{}_ex_{}_speaker_{}.wav'.format(example_dir, epoch, i,s))
                    utils.reconstruct_wav_from_spec(mixed_spec[i],
                                                    '{}/mixed_spec_example_epoch_{}_ex_{}.wav'.format(example_dir, epoch, i))


        total_loss = 0
        sdr_db = 0
        for i, sample in tqdm(enumerate(loader), total=len(loader), desc='Evaluation'):
            targets = sample['clean_spec'].cuda()
            face_embs, mixed_spec = sample['face_embeds'].cuda(), sample['mixed_spec'].cuda()
            outputs = net(face_embs, mixed_spec)
            loss = loss_criterion(outputs, targets)
            sdr_db += utils.calculate_SDR(outputs,targets)
            total_loss += loss.item()
        total_loss /= len(loader.dataset)
        sdr_db /= len(loader.dataset)

        logger.write_to_log('{}, {}'.format(epoch, total_loss), 'eval_loss.csv')
        logger.write_to_log('{}, {}'.format(epoch, sdr_db), 'eval_sdr.csv')

    net.train()


def train_step(net, optimizer, loss_criterion, sample, args):
    optimizer.zero_grad()
    targets = sample['clean_spec'].cuda()
    face_embs, spec = sample['face_embeds'].cuda(), sample['mixed_spec'].cuda()

    if args.n_speaker == 1:
        outputs = net(face_embs, spec)
        loss = loss_criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    else:
        if not args.audio_only:
            loss = torch.zeros(1).cuda()
            for perm in [(0,1),(1,0)]:
                outputs = net(face_embs[:, perm], spec)
                loss += loss_criterion(outputs, targets[:,perm])
            loss.backward()
            optimizer.step()
        else:
            loss = torch.zeros(1).cuda()
            for sample_idx in range(spec.shape[0]):
                outputs = net(None, spec[sample_idx].unsqueeze(0))
                loss_0 = loss_criterion(outputs, targets[sample_idx,(0,1)].unsqueeze(0))
                loss_1 = loss_criterion(outputs, targets[sample_idx,(1,0)].unsqueeze(0))
                min_loss = loss_0 if loss_0 < loss_1 else loss_1
                min_loss.backward()
            optimizer.step()
            loss += min_loss
            outputs = None


    return outputs, loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--eval')
    parser.add_argument('--root_dir','-root')
    parser.add_argument('--n_speaker', type=int)
    parser.add_argument('--mix',type=float)
    parser.add_argument('--suffix', default='')
    parser.add_argument('--interference_eval','-ie')
    parser.add_argument('--interference_train','-it')
    parser.add_argument('--interference_root_dir','-iroot')
    parser.add_argument('--gradual',action='store_true')
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--audio_only',action='store_true',help='Removes video stream network')

    train(parser.parse_args())
