import librosa
import soundfile as sf
import pyrubberband as pyrb
from scheme import extract_all_sfps, get_BER, get_mel_spec, compare_whole_signal, extract_shifted_fps

AUDIO_DIR = 'audio_files'
# parameters for extract_shifted_fps() in scheme.py
ST_OFFSET = 1.6
STEP = 0.8

def fingerprint_hit(BER):
    return BER <= 0.35

def float_filename(f : float) -> str:
    """
    For a given float, returns its corresponding string
    with '-' replacing any '.'
    Used to create a valid filename.
    """
    r = ''
    s = str(f)
    for char in s:
        if char == '.':
            r += '-'
        else:
            r += char
    return r

def pitch_attack_simple(audioname: str, n_sts: float, write: bool) -> float:
    """
    Pitch shift entire signal. Uses simple_fingerprinting: calculate BER between fingerprint of 
    original audio and of pitch shifted audio
    @param audio: Original audio digital signal
    @param sr: Audio's sample rate
    @param n_steps: Number of semitones to raise the audio's pitch
    @param write: Whether to write to a file or not

    Returns:
    Bit error rate (0 <= BER <= 1) of entire signal between subfingerprints of the original audio and the shifted audio.
    """
    x, sr = librosa.load(f'{AUDIO_DIR}/{audioname}/{audioname}.wav')
    x_shift = pyrb.pitch_shift(x, sr, n_sts)

    if write:
        sf.write(f'{AUDIO_DIR}/{audioname}/{audioname}{float_filename(n_sts)}.wav', x_shift, sr)

    original_spec = get_mel_spec(x, sr)
    shifted_spec = get_mel_spec(x_shift, sr)

    return compare_whole_signal(original_spec, shifted_spec)

def least_BER(BERs: dict):
    return min(BERs.values())

def pitch_attack_robust(audioname: str, n_sts: float, write: bool) -> dict:
    """
    Pitch shift entire signal. Uses pitch-robust fingerprinting scheme.
    This fingerprinting scheme consist of computing not only the fingerprint
    of the original signal but also as computing fingerprints 
    of pitch shifted versions of the signal by varying amounts.
    See extract_shifted_fps for the details on how much is the signal 
    pitch shifted by for each fingerprint
    
    @param audioname: Do not include file extension
    @param n_sts: Number of semitones to pitch shift the input audio by
    @param write: Whether to write the pitch-shifted audio to a wav file

    Returns: A BER dictionary: each entry is the Bit Error Rate between
    the fingerprint for our pitch_shifted input audio and one of the 
    fingerprints in the robust fingerprint dictionary, indexed by the amount
    of pitch shifting applied to the fingerprint in the dictionary
    
    """
    x, sr = librosa.load(f'{AUDIO_DIR}/{audioname}/{audioname}.wav')
    x_shift = pyrb.pitch_shift(x, sr, n_sts)
    if write:
        sf.write(f'{AUDIO_DIR}/{audioname}/{audioname}{float_filename(n_sts)}.wav', x_shift, sr)

    x_shift_sfps = extract_all_sfps(get_mel_spec(x_shift, sr))

    fp_list = extract_shifted_fps(x, sr, ST_OFFSET, STEP)
    BERs = {}
    for st in fp_list.keys():
        BER = round(get_BER(x_shift_sfps, fp_list[st]), 3)
        BERs[st] = BER
    return BERs


