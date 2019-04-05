from torch import nn
import torch.nn.functional as F
import torch

MOMENTUM = 0.10

def SamePadConv1d(in_channels, out_channels, kernel_size, dilation=1):
    pad = (dilation * (kernel_size - 1)) // 2
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation,
                     padding=pad)


def SamePadConv2d(in_channels, out_channels, kernel_size, dilation=(1, 1)):
    pad = [
        (dilation[0] * (kernel_size[0] - 1)) // 2,
        (dilation[1] * (kernel_size[1] - 1)) // 2
    ]
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation,
                     padding=pad)


class AudioNet(nn.Module):
    """
    NCFT
    """
    def __init__(self):
        super(AudioNet, self).__init__()
        self.conv_1 = SamePadConv2d(in_channels=2, out_channels=96, kernel_size=(7, 1))
        self.conv_2 = SamePadConv2d(in_channels=96, out_channels=96, kernel_size=(1, 7))
        self.conv_3 = SamePadConv2d(in_channels=96, out_channels=96, kernel_size=(5, 5))
        self.conv_4 = SamePadConv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=(1, 2))
        self.conv_5 = SamePadConv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=(1, 4))
        self.conv_6 = SamePadConv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=(1, 8))
        self.conv_7 = SamePadConv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=(1, 16))
        self.conv_8 = SamePadConv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=(1, 32))
        self.conv_9 = SamePadConv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=(1, 1))
        self.conv_10 = SamePadConv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=(2, 2))
        self.conv_11 = SamePadConv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=(4, 4))
        self.conv_12 = SamePadConv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=(8, 8))
        self.conv_13 = SamePadConv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=(16, 16))
        self.conv_14 = SamePadConv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), dilation=(32, 32))
        self.conv_15 = SamePadConv2d(in_channels=96, out_channels=8, kernel_size=(1, 1), dilation=(1, 1))

        self.batch_norm_1 = nn.BatchNorm2d(num_features=96, momentum=MOMENTUM, eps=1e-5)
        self.batch_norm_2 = nn.BatchNorm2d(num_features=96, momentum=MOMENTUM, eps=1e-5)
        self.batch_norm_3 = nn.BatchNorm2d(num_features=96, momentum=MOMENTUM, eps=1e-5)
        self.batch_norm_4 = nn.BatchNorm2d(num_features=96, momentum=MOMENTUM, eps=1e-5)
        self.batch_norm_5 = nn.BatchNorm2d(num_features=96, momentum=MOMENTUM, eps=1e-5)
        self.batch_norm_6 = nn.BatchNorm2d(num_features=96, momentum=MOMENTUM, eps=1e-5)
        self.batch_norm_7 = nn.BatchNorm2d(num_features=96, momentum=MOMENTUM, eps=1e-5)
        self.batch_norm_8 = nn.BatchNorm2d(num_features=96, momentum=MOMENTUM, eps=1e-5)
        self.batch_norm_9 = nn.BatchNorm2d(num_features=96, momentum=MOMENTUM, eps=1e-5)
        self.batch_norm_10 = nn.BatchNorm2d(num_features=96, momentum=MOMENTUM, eps=1e-5)
        self.batch_norm_11 = nn.BatchNorm2d(num_features=96, momentum=MOMENTUM, eps=1e-5)
        self.batch_norm_12 = nn.BatchNorm2d(num_features=96, momentum=MOMENTUM, eps=1e-5)
        self.batch_norm_13 = nn.BatchNorm2d(num_features=96, momentum=MOMENTUM, eps=1e-5)
        self.batch_norm_14 = nn.BatchNorm2d(num_features=96, momentum=MOMENTUM, eps=1e-5)
        self.batch_norm_15 = nn.BatchNorm2d(num_features=8, momentum=MOMENTUM, eps=1e-5)

    def forward(self, x):
        x = self.batch_norm_1(F.relu(self.conv_1(x)))
        x = self.batch_norm_2(F.relu(self.conv_2(x)))
        x = self.batch_norm_3(F.relu(self.conv_3(x)))
        x = self.batch_norm_4(F.relu(self.conv_4(x)))
        x = self.batch_norm_5(F.relu(self.conv_5(x)))
        x = self.batch_norm_6(F.relu(self.conv_6(x)))
        x = self.batch_norm_7(F.relu(self.conv_7(x)))
        x = self.batch_norm_8(F.relu(self.conv_8(x)))
        x = self.batch_norm_9(F.relu(self.conv_9(x)))
        x = self.batch_norm_10(F.relu(self.conv_10(x)))
        x = self.batch_norm_11(F.relu(self.conv_11(x)))
        x = self.batch_norm_12(F.relu(self.conv_12(x)))
        x = self.batch_norm_13(F.relu(self.conv_13(x)))
        x = self.batch_norm_14(F.relu(self.conv_14(x)))
        x = self.batch_norm_15(F.relu(self.conv_15(x)))

        return x


class VideoNet(nn.Module):

    def __init__(self, n_time_bins, embedding_len):
        super(VideoNet, self).__init__()
        self.n_time_bins = n_time_bins
        self.conv_1 = SamePadConv1d(in_channels=embedding_len, out_channels=256, kernel_size=7)
        self.conv_2 = SamePadConv1d(in_channels=256, out_channels=256, kernel_size=5)
        self.conv_3 = SamePadConv1d(in_channels=256, out_channels=256, kernel_size=5, dilation=2)
        self.conv_4 = SamePadConv1d(in_channels=256, out_channels=256, kernel_size=5, dilation=4)
        self.conv_5 = SamePadConv1d(in_channels=256, out_channels=256, kernel_size=5, dilation=8)
        self.conv_6 = SamePadConv1d(in_channels=256, out_channels=256, kernel_size=5, dilation=16)

        self.batch_norm_1 = nn.BatchNorm1d(num_features=256, momentum=MOMENTUM, eps=1e-5)
        self.batch_norm_2 = nn.BatchNorm1d(num_features=256, momentum=MOMENTUM, eps=1e-5)
        self.batch_norm_3 = nn.BatchNorm1d(num_features=256, momentum=MOMENTUM, eps=1e-5)
        self.batch_norm_4 = nn.BatchNorm1d(num_features=256, momentum=MOMENTUM, eps=1e-5)
        self.batch_norm_5 = nn.BatchNorm1d(num_features=256, momentum=MOMENTUM, eps=1e-5)
        self.batch_norm_6 = nn.BatchNorm1d(num_features=256, momentum=MOMENTUM, eps=1e-5)

    def forward(self, x):
        x = self.batch_norm_1(F.relu(self.conv_1(x)))
        x = self.batch_norm_2(F.relu(self.conv_2(x)))
        x = self.batch_norm_3(F.relu(self.conv_3(x)))
        x = self.batch_norm_4(F.relu(self.conv_4(x)))
        x = self.batch_norm_5(F.relu(self.conv_5(x)))
        x = self.batch_norm_6(F.relu(self.conv_6(x)))
        x = F.interpolate(x, size=self.n_time_bins, mode='nearest')

        return x


class BigNet(nn.Module):

    def __init__(self, n_time_bins,embedding_len,audio_only=False):
        super(BigNet, self).__init__()
        self.n_time_bins = n_time_bins
        self.audio_only = audio_only

        self.video_net = VideoNet(self.n_time_bins,embedding_len)
        self.audio_net = AudioNet()
        if not self.audio_only:
            self.blstm = nn.LSTM(input_size=257 * 8 + 256, hidden_size=200, num_layers=1, batch_first=True,
                                 bidirectional=True)
        else:
            self.blstm = nn.LSTM(input_size=257 * 8, hidden_size=200, num_layers=1, batch_first=True,
                                 bidirectional=True)

        self.fc_1 = nn.Linear(400, 600)
        self.fc_2 = nn.Linear(600, 600)
        self.fc_3 = nn.Linear(600, 2 * 257)

    def _complex_mult(self, mask, spectrogram):
        mask_real = mask[:, 0:1]
        mask_imag = mask[:, 1:2]
        spectrogram_real = spectrogram[:, 0:1]
        spectrogram_imag = spectrogram[:, 1:2]
        clean_spectrogram = torch.cat([mask_real * spectrogram_real - mask_imag * spectrogram_imag,
                                       mask_imag * spectrogram_real + spectrogram_imag * mask_real], dim=1)
        return clean_spectrogram

    def forward(self, face_embeds, spectrogram,return_mask=False):
        """
        face_embeds: BATCH_SIZE * N_FEATURES * TIME_BINS_IN_VIDEO_FRAMES (NCT)
        spectrogram: BATCH_SIZE * N_CHANNELS(2,real and imag) *F_BINS * TIME_BINS (NCFT)
        :param face_embeds:
        :param spectrogram:
        :return:
        """
        audio_stream = self.audio_net(spectrogram) # NCFT
        audio_stream = audio_stream.view(-1,  257 * 8, self.n_time_bins)  # NCFT - > N(CxF)T
        # print(torch.equal(audio_stream[:,:,2,:].view(1,-1), audio_stream_cp[:,:,2].view(1,-1)))
        if not self.audio_only:
            video_stream = self.video_net(face_embeds) # NCT
            fusion = torch.cat((audio_stream, video_stream), dim=1).transpose(1,2)  # batch,channel,seq(time)->batch,seq(time),channels
        else:
            fusion = audio_stream.transpose(1,2)  # batch,channel,seq(time)->batch,seq(time),channels

        # TODO: See if this flatten params ok
        self.blstm.flatten_parameters()
        fusion, (hidden_state, cell_state) = self.blstm(fusion)
        fusion = F.relu(fusion)
        fusion = F.relu(self.fc_1(fusion))
        fusion = F.relu(self.fc_2(fusion))
        fusion = torch.sigmoid(self.fc_3(fusion))
        mask = fusion.view(-1, self.n_time_bins, 257, 2).transpose(1,3) # NTFC->NCFT
        clean_spectrogram = self._complex_mult(mask, spectrogram)

        if return_mask:
            return clean_spectrogram, mask
        else:
            return clean_spectrogram


class DoubleSpeakerNet(nn.Module):

    def __init__(self, n_time_bins, embedding_len,n_speakers=1,audio_only=False):
        super(DoubleSpeakerNet, self).__init__()
        self.n_time_bins = n_time_bins
        self.n_speakers = n_speakers
        self.audio_only = audio_only

        self.video_net = VideoNet(self.n_time_bins,embedding_len)
        self.audio_net = AudioNet()
        if not self.audio_only:
            self.blstm = nn.LSTM(input_size=257 * 8 + 256*self.n_speakers, hidden_size=200, num_layers=1, batch_first=True,
                                 bidirectional=True)
        else:
            self.blstm = nn.LSTM(input_size=257 * 8, hidden_size=200, num_layers=1,
                                 batch_first=True,
                                 bidirectional=True)

        self.fc_1 = nn.Linear(400, 600)
        self.fc_2 = nn.Linear(600, 600)
        self.fc_3 = nn.Linear(600, 2 * 257 * self.n_speakers    )

    def _complex_mult(self, mask, spectrogram):
        mask_real = mask[:,  0:1]
        mask_imag = mask[:,  1:2]
        spectrogram_real = spectrogram[:, 0:1]
        spectrogram_imag = spectrogram[:, 1:2]
        clean_spectrogram = torch.cat([mask_real * spectrogram_real - mask_imag * spectrogram_imag,
                                       mask_imag * spectrogram_real + spectrogram_imag * mask_real], dim=1)
        return clean_spectrogram

    def forward(self, face_embed_batch, spectrogram, return_mask = False):
        """
        face_embeds: BATCH_SIZE * N_FEATURES * TIME_BINS_IN_VIDEO_FRAMES (NSCT)
        spectrogram: BATCH_SIZE * N_CHANNELS(2,real and imag) *F_BINS * TIME_BINS (NCFT)
        :param face_embeds:
        :param spectrogram:
        :return: NSCFT
        """
        audio_stream = self.audio_net(spectrogram)  # NCFT
        audio_stream = audio_stream.view(-1, 257 * 8, self.n_time_bins)  # NCFT - > N(CxF)T
        if not self.audio_only:
            video_stream = torch.cat([self.video_net(face_embed_batch[:,i]) for i in range(self.n_speakers)],dim=1) # N(CxS)T
            fusion = torch.cat((audio_stream, video_stream), dim=1).transpose(1, 2)  # batch,channel,seq(time)->batch,seq(time),channels (NCT->NTC)
        else:
            fusion = audio_stream.transpose(1, 2)
        # TODO: See if this flatten params ok
        self.blstm.flatten_parameters()
        fusion, (hidden_state, cell_state) = self.blstm(fusion)
        fusion = F.relu(fusion)
        fusion = F.relu(self.fc_1(fusion))
        fusion = F.relu(self.fc_2(fusion))
        fusion = torch.sigmoid(self.fc_3(fusion))  # NTC
        mask = fusion.view(-1, self.n_time_bins,self.n_speakers, 257, 2).permute(0,2,4,3,1)  # NTC->NTFC->NCFT, NTC->NTSFC -> NSCFT
        clean_spectrogram = torch.stack([self._complex_mult(mask[:,i], spectrogram) for i in range(self.n_speakers)],dim=1)

        if return_mask:
            return clean_spectrogram, mask
        else:
            return clean_spectrogram


