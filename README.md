# Evaluating Audiovisual Source Separation in the Context of Video Conferencing

# Installation
- Extract 'model_weights' directory into root of this repo. It is not included in the repo. Contact me if you need it.
- Install dependencies
```bash
conda install numpy
conda install cmake # if you don't have it
conda install pytorch  # v0.4.2 for this version
conda install -c conda-forge opencv
conda install -c conda-forge librosa 
pip install dlib 
pip install pystoi # These two are not necessary for demo, may remove their dependencies later
pip install mir_eval
```
- There might be more dependencies you might need, try using `conda` or `pip` in a similar way.
# Running demo

Demo requires a noisy video as input, longer than 3 seconds, 25 fps, and only one face is detectable whole time.
If the video contains multiple faces, the workaround is to crop your video to only contain your target speaker.

If you want to create synthetic input videos, you can use:
```bash
ffmpeg -i target_video.mp4 -i interfering_video.mp4 -c:v copy -map 0:v:0 -map 1:a:0 mixed_video.mp4
```
If you have only audio for interfere, you can use,
```bash
ffmpeg -i target_video.mp4 -i interfering_audio.wav -c:v copy -map 0:v:0 -map 1:a:0 mixed_video.mp4
```

```bash
python run_on_video.py --model [PATH/TO/MODEL.pth]
                       --video_path [PATH/TO/INPUT_VIDEO.mp4]
                       --audio_out [OUTPUT/PATH/TO/DENOISED/AUDIO.wav]
                       --video_out [OUTPUT/PATH/TO/DENOISED/VIDEO.wav]
```

Models can be found under `model_weights/separation_models`.

# Training

Training only works on preprocessed data for efficiency.
Check [this repo](https://github.com/berkayinan/avspeech_dl) if you need to download and preprocess the dataset.

After you finish, you should have two directories. One for video embeddings, and one for corresponding audio spectrograms.
Right now, there is some automatic way to find the corresponding audio for each video, but it is hardcoded.
Then you can run,

```bash
python train.py --train [/PATH/TO/CSV/FILE/FOR/TRAINING/VIDEOS]
                --eval [/PATH/TO/CSV/FILE/FOR/EVALUATION/VIDEOS]
                --root_dir [/PATH/TO/VIDEO/EMBEDDINGS]
                --interference_train [/PATH/TO/CSV/FILE/FOR/TRAINING/VIDEOS/TO/INTERFERE] # Interfere = noise
                --interference_eval [/PATH/TO/CSV/FILE/FOR/EVALUATION/VIDEOS/TO/INTERFERE]
                --interference_root_dir [/PATH/TO/VIDEO/EMBEDDINGS/TO/INTERFERE]
                --mix [FLOAT COEFFCIENT] # number to multiply noise amplitude before mixing
                --suffix [A REMINDER TEXT TO ADD TO LOGS, NAME OF EXPERIMENT]
                --gradual # experimental, ignore if not testing. Increases noise level during training
                --lr [LEARNING RATE] # default=3e-4
                --n_speaker [NUMBER OF SPEAKER FACES] # This should be 1. 2 is experimental.
                --audio_only # Removes video stream network'
```

Logs and models will be saved on `~/logs/`.
You can plot the training process by using the csv files written under. They are updated before each epoch.