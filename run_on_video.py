import torch
import argparse
import model
import numpy as np
import utils
import librosa
from lipreading.utils import get_lipreading_features

def run_model_on_embeddings(net, video_embeddings, audio_spectrogram, output_path):
    """
    :param video_embeddings:  [Frame count x embedding length] torch tensor
    :param audio_spectrogram: [257 x time bins] numpy matrix
    :return:
    """
    with torch.no_grad():
        full_len = video_embeddings.shape[0] - video_embeddings.shape[0] % 75
        out_wavs = []
        for frame_start in range(0, full_len, 75):
            print('-- Window: {}/{}'.format(frame_start,full_len))
            cropped_face_embeds = video_embeddings[frame_start:frame_start + 75].transpose(0, 1)
            spec_start = frame_start * 4
            cropped_original_spec = torch.from_numpy(audio_spectrogram[:, :, spec_start:spec_start + 301])
            cropped_original_spec = utils.apply_power_compression(cropped_original_spec)
            assert cropped_original_spec.shape == (2, 257, 301)
            out = net(cropped_face_embeds.unsqueeze(0),
                      cropped_original_spec.unsqueeze(0),
                      False)
            out = out.cpu().detach().numpy()
            out_wavs.append(utils._get_wav_np_from_spec(out[0], power_law=True))
        print('-- Window: {}/{}'.format(full_len, full_len))

        stacked = np.hstack(out_wavs)
        librosa.output.write_wav(output_path, stacked, 16000)


def load_bignet():
    net = torch.nn.DataParallel(model.BigNet(301, 512))
    print('opening file')
    if torch.cuda.is_available():
        weights = torch.load(args.model)
        net.cuda()
    else:
        weights = torch.load(args.model, map_location='cpu')
    net.load_state_dict(weights)
    net.eval()
    print('weights loaded')
    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to model weights file to load')
    parser.add_argument('--video_path', help='Path to video file')
    parser.add_argument('--audio_out', help='Path for output wav file')
    parser.add_argument('--video_out', default=None, help='[optional] Path for output video file')

    args = parser.parse_args()

    n_steps = 4 if args.video_out is None else 5
    print('[1/{}] Loading fusion network'.format(n_steps))
    net = load_bignet()

    print('[2/{}] Extracting lip reading embeddings'.format(n_steps))
    lip_reading_embeddings = get_lipreading_features(args.video_path, None, False)

    print('[3/{}] Computing audio spectrogram'.format(n_steps))

    audio_spectrogram = utils._get_spec(args.video_path)

    print('[4/{}] Running Fusion network on embeddings and spectogram'.format(n_steps))
    run_model_on_embeddings(net, lip_reading_embeddings, audio_spectrogram, output_path=args.audio_out)

    if args.video_out is not None:
        print('[5/{}] Mergining output audio to original video'.format(n_steps))
        utils.merge_audio_with_video(args.video_path,args.audio_out, args.video_out)
