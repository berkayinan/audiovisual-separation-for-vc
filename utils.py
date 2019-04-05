import librosa
import torch
import numpy as np
import os
import time
import mir_eval
import subprocess
import multiprocessing
from pystoi import stoi
import tempfile
import subprocess
EMBEDDING_LENS = {'openface': 736,
                  'lipreading': 512,
                  'facenet': 1792}
ROOT_DIRS = {'openface': '/home/berkay/fastdata/AVSpeech/openface',
             'lipreading': '/home/berkay/fastdata/AVSpeech/lipreading_512',
             'facenet': '/home/berkay/fastdata/AVSpeech/train'}


class SingletonPool:
    pool = None

    @staticmethod
    def get_pool():
        if SingletonPool.pool is None:
            SingletonPool.pool = multiprocessing.Pool(6)
        return SingletonPool.pool

    @staticmethod
    def close_pool():
        if SingletonPool.pool is not None:
            print('Closing pool')
            SingletonPool.pool.close()
            SingletonPool.pool.join()
            SingletonPool.pool = None


def video_name_from_row(row):
    video_output_filename = row['youtube_id'] + '_time_' + row['time_start'] + '-' + row['time_end'] + '.mp4'
    return video_output_filename


def reconstruct_wav_from_spec(spec, output_path, power_law=True):
    rec_sig_crop = _get_wav_np_from_spec(spec, power_law)
    librosa.output.write_wav(output_path, rec_sig_crop, 16000)


def _get_wav_np_from_spec(spec, power_law=True):
    if type(spec) == np.ndarray:
        if power_law:
            spec = np.abs(spec) ** (1 / 0.3) * np.sign(spec)
    else:
        if power_law:
            spec = spec.abs().pow(1 / 0.3) * spec.sign()
        spec = spec.cpu().detach().numpy()

    spec = spec[0] + 1j * spec[1]
    rec_sig_crop = librosa.istft(spec, hop_length=10 * 16, win_length=25 * 16, center=True, length=16000 * 3)
    return rec_sig_crop


def _get_spec(video_path):
    signal, sr = librosa.load(video_path, sr=16000)
    spectrogram = librosa.core.stft(signal,
                                    n_fft=512,
                                    hop_length=10 * 16,
                                    win_length=25 * 16,
                                    window='hann',
                                    center=True)
    spectrogram_sep = np.stack([spectrogram.real, spectrogram.imag])
    print(spectrogram_sep.shape)
    return spectrogram_sep


def load_arbitrary_data():
    face_embeds = torch.from_numpy(np.load('helmut_noise.npy')[:3 * 25, :].T)
    noisy_spec = torch.from_numpy(_get_spec('helmut_solo.mp4_noisy_kb.mp4_merged.mp4')[:, :, :301])
    clean_spec = torch.from_numpy(_get_spec('helmut_solo.mp4')[:, :, :301])
    noisy_spec = apply_power_compression(noisy_spec)
    clean_spec = apply_power_compression(clean_spec)
    print(face_embeds.shape, noisy_spec.shape, clean_spec.shape)
    return {'face_embeds': face_embeds, 'mixed_spec': noisy_spec, 'clean_spec': clean_spec, 'clean_video_info': {},
            'interference_video_info': {}}


def apply_power_compression(spectrogram):
    """
    Pytorch doesn't allow fractional exponents with negative base. So we get abs before pow and multiply with sign.
    :param spectrogram:
    :return:
    """
    return spectrogram.abs().pow(0.3) * spectrogram.sign()


def load_face_embeds(row, input_dir):
    video_name = video_name_from_row(row)
    output_path = os.path.join(input_dir, 'face_embeds', video_name[:2], video_name[:-4]) + '.npy'
    return np.load(output_path)


def load_spectrogram(row, input_dir):
    video_name = video_name_from_row(row)
    if 'openface' in input_dir or 'lipreading' in input_dir:
        audio_path = os.path.join('/home/berkay/fastdata/AVSpeech/train', 'specs', video_name[:2],
                                  video_name[:-4]) + '.npy'
    else:
        audio_path = os.path.join(input_dir, 'specs', video_name[:2], video_name[:-4]) + '.npy'
    return np.load(audio_path)


def calculate_SDR(outputs, targets):
    def _get_sdr(output, target):
        pred = np.expand_dims(_get_wav_np_from_spec(output), 0)
        reference = np.expand_dims(_get_wav_np_from_spec(target), 0)
        sdr, _, _, _ = mir_eval.separation.bss_eval_sources(pred, reference, False)
        return sdr

    total_sdr = 0
    if len(targets.shape) == 4:
        for output, target in zip(outputs, targets):
            total_sdr += _get_sdr(output, target)
    else:
        for output_speakers, target_speakers in zip(outputs, targets):
            for output, target in zip(output_speakers, target_speakers):
                total_sdr += _get_sdr(output, target) / 2
    return total_sdr[0]


def _pesq_for_multiprocessing(output, target, idx):
    idx = next(tempfile._get_candidate_names())
    PESQ_PATH = '/home/berkay/pesq/source/PESQ'


    output_wav_raw = '/tmp/pesq_temp/evaltest_output_{}.wav'.format(idx)
    target_wav_raw = '/tmp/pesq_temp/evaltest_target_{}.wav'.format(idx)

    reconstruct_wav_from_spec(output, output_wav_raw)
    reconstruct_wav_from_spec(target, target_wav_raw)

    # Convert precision to 16bit PCM
    output_wav_16bits = '/tmp/pesq_temp/evaltest_output_16_{}.wav'.format(idx)
    target_wav_16bits = '/tmp/pesq_temp/evaltest_target_16_{}.wav'.format(idx)
    try:
        subprocess.run(['sox', output_wav_raw, '-b', '16', output_wav_16bits], check=True)
        subprocess.run(['sox', target_wav_raw, '-b', '16', target_wav_16bits], check=True)

        shell_output = subprocess.run(
            [PESQ_PATH, '+16000', '+wb', target_wav_16bits, output_wav_16bits], check=True,
            stdout=subprocess.PIPE).stdout
    except subprocess.CalledProcessError as e:
        print(str(e))
        print(e.output)
        print(e.stdout)
        print(e.stderr)
        return 0

    shell_output = shell_output.decode()
    # pesq_val = float(shell_output.splitlines()[-1].split(' ')[-1].split('\t')[0].strip())
    pesq_val = float(shell_output.splitlines()[-1].split(' ')[-1])


    print(pesq_val)

    for temp_file  in [output_wav_raw,target_wav_raw,output_wav_16bits,target_wav_16bits]:
        os.remove(temp_file)
    return pesq_val


def get_pesq(outputs, targets):
    outputs = outputs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    pool = SingletonPool.get_pool()
    pesq_res = pool.starmap(_pesq_for_multiprocessing, zip(outputs, targets, range(targets.shape[0])))
    return sum(pesq_res)


def _cdist_from_files(output_file, target_file, sample_number):
    return_text = subprocess.run(['/home/berkay/phonvoc/cdist.sh', target_file, output_file, str(sample_number)],
                                 stdout=subprocess.PIPE).stdout
    return float(return_text.strip())


def _get_cdist(output, target, sample_number):
    output_filename = '/tmp/pred_cdist_eval_{}.wav'.format(sample_number)
    target_filename = '/tmp/target_cdist_eval_{}.wav'.format(sample_number)
    reconstruct_wav_from_spec(output, output_filename)
    reconstruct_wav_from_spec(target, target_filename)
    cdist = _cdist_from_files(target_filename, output_filename, sample_number)
    os.remove(output_filename)
    os.remove(target_filename)
    return cdist


def get_cep_distance(outputs, targets):
    pool = SingletonPool.get_pool()
    outputs = outputs.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    if len(targets.shape) == 4:
        cdist_array = pool.starmap(_get_cdist, zip(outputs, targets, range(len(outputs))))

        # for output, target in zip(outputs, targets):
        #     total_cdist += _get_cdist(output, target)
    else:
        raise NotImplementedError
        # for output_speakers, target_speakers in zip(outputs, targets):
        #     for output, target in zip(output_speakers, target_speakers):
        #         total_cdist += _get_cdist(output, target)
    return sum(cdist_array)



def get_SNR(target_signals,noise_signals):
    def _get_spec_from_torch_spec(torchspec):
        reverse_plaw = torchspec.abs().pow(1 / 0.3) * torchspec.sign()
        np_spec = reverse_plaw.cpu().detach().numpy()

        np_spec = np_spec[0] + 1j * np_spec[1]
        return np_spec

    total_SNR = 0
    for target_spec,noise_spec in zip(target_signals,noise_signals):
        np_target = _get_spec_from_torch_spec(target_spec)
        p_target = np.sqrt(np.sum(np.abs(np_target)**2))
        p_noise = np.sqrt(np.sum(np.abs(_get_spec_from_torch_spec(noise_spec)-np_target)**2))
        snr_sample = 20*np.log10(p_target/p_noise)
        if snr_sample != np.inf:
            total_SNR += snr_sample
    return total_SNR


def get_STOI(outputs,targets):
    total_STOI = 0
    for output, target in zip(outputs,targets):
        total_STOI += stoi.stoi(
            _get_wav_np_from_spec(target),
            _get_wav_np_from_spec(output),
            16000,
            extended=False)
    return total_STOI


def merge_audio_with_video(video_path,audio_path,
                           merged_video_output_path):
    subprocess.run(['ffmpeg', '-y',
                    '-i', video_path,
                    '-i', audio_path,
                    '-c:v', 'copy',
                    '-map', '0:v:0',
                    '-map', '1:a:0',
                    merged_video_output_path],
                    stderr=subprocess.PIPE)


class MyLogger:
    def __init__(self, suffix):
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.logs_dir = os.path.join(os.path.expanduser('~'), 'logs', 'logs_{}_{}'.format(self.timestamp, suffix))
        os.makedirs(self.logs_dir, exist_ok=False)
        os.makedirs(os.path.join(self.logs_dir, 'models'))
        self.log_backup = {}

    def write_to_log(self, line, file_path):
        if file_path not in self.log_backup:
            self.log_backup[file_path] = []
        self.log_backup[file_path].append(line)
        with open(os.path.join(self.logs_dir, file_path), 'a') as file:
            file.write(line + '\n')
