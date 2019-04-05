import torch
import argparse
import model
import numpy as np
import utils
import librosa
import mir_eval
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--video')

    args = parser.parse_args()
    net = torch.nn.DataParallel(model.BigNet(301,512))
    print('opening file')

    if torch.cuda.is_available():
        weights = torch.load(args.model)
        net.cuda()
    else:
        weights = torch.load(args.model,map_location='cpu')

    net.load_state_dict(weights)
    net.eval()

    #--model /home/berkay/logs/logs_20181010_150915_looooong/models/model_epoch_171_loss_0.pth
    #home/berkay/logs/logs_20181024_181102_speaker_and_noise/models/model_epoch_33_loss_0.pth
    print('weights loaded')

    if args.video is None:
        original_face_embeds = np.load('sample_embeddings/helmut_lipreading_features.npy')

        # original_face_embeds = np.load('helmut_embeddings.npy')
        # original_face_embeds = np.zeros_like(original_face_embeds)

        original_spectrogram = utils._get_spec('video_files/helmut_nogap_boosted.mp4')
        # original_spectrogram = utils._get_spec('helmut_solo.mp4_noisy_kb.mp4_merged.mp4')
        # original_spectrogram = utils._get_spec('helmut_feifei.mp4')
        
        gt_spectrogram = utils._get_spec('video_files/celal_sengor.mp4')
        # inf_spectrogram = utils._get_spec('feifei_crop.mp4')

        full_len = original_face_embeds.shape[0] - original_face_embeds.shape[0] % 75
        out_wavs = []
        for frame_start in range(0,full_len,75):
            cropped_face_embeds = torch.from_numpy(original_face_embeds[frame_start:frame_start + 75]).transpose(0, 1)
            spec_start = frame_start*4
            cropped_original_spec = torch.from_numpy(original_spectrogram[:, :, spec_start:spec_start + 301])
            cropped_original_spec = utils.apply_power_compression(cropped_original_spec)
            print(cropped_face_embeds.shape)
            # assert cropped_face_embeds.shape == (1792, 75) or cropped_face_embeds.shape == (736,75)
            assert cropped_original_spec.shape == (2, 257, 301)
            out, mask = net(cropped_face_embeds.unsqueeze(0),
                           cropped_original_spec.unsqueeze(0),
                           True)
            out = out.cpu().detach().numpy()
            mask = mask.cpu().detach().numpy()
            np.save('/home/berkay/thesis_files/mask_{}.npy'.format(frame_start),mask)
            utils.reconstruct_wav_from_spec(out[0],'/tmp/ss_{}.wav'.format(frame_start))
            out_wavs.append(utils._get_wav_np_from_spec(out[0]))

            cropped_gt = gt_spectrogram[:, :, spec_start:spec_start + 301]
            np.save('/home/berkay/thesis_files/noisy_{}.npy'.format(frame_start),cropped_gt)

            np.save('/home/berkay/thesis_files/denoised_{}.npy'.format(frame_start),np.abs(out[0]) ** (1 / 0.3) * np.sign(out[0]))

            pred_samples = utils._get_wav_np_from_spec(out[0])
            ref_samples = utils._get_wav_np_from_spec(cropped_gt,power_law=False)
            # inf_samples = utils._get_wav_np_from_spec(inf_spectrogram[:, :, spec_start:spec_start + 301])

            print(mir_eval.separation.bss_eval_sources(np.array([ref_samples]),np.array([pred_samples]),False)[0][0])

        stacked =np.hstack(out_wavs)
        print(stacked.shape)
        # librosa.output.write_wav('/tmp/helmut_vs_lr_output.wav',stacked , 16000)

