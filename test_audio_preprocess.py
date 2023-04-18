import librosa
import os
import soundfile as sf


def preprocess_test_audio():
    out_dir = r'test_audio'
    os.makedirs(out_dir, exist_ok=True)
    audio_file = r'full_test/soundscape_29201.ogg'
    wave, sr = librosa.load(audio_file)
    segment_dur_secs = 5
    segment_length = sr * segment_dur_secs
    split = []
    for s in range(0, len(wave), segment_length):
        t = wave[s: s + segment_length]
        split.append(t)
    recording_name = 'row'
    for i, segment in enumerate(split):
        out_file = f"{i}.ogg"
        sf.write(os.path.join(out_dir, out_file), segment, sr)



if __name__ == '__main__':
    preprocess_test_audio()