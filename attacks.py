import librosa
import soundfile as sf
import pyrubberband as pyrb
from scheme import get_fp, get_BER, get_mel_spec, compare_whole_signal, get_robust_fps

AUDIO_DIR = 'audio_files'
BER_THRESHOLD = 0.35
# parameters for get_robust_fps() in scheme.py
ST_OFFSET = 1.6
STEP = 0.8

def fingerprint_hit(BER):
    return BER <= BER_THRESHOLD

def float_filename(f : float) -> str:
    """
    For a given float, returns its corresponding string
    with '-' replacing any '.'
    Used to write a valid filename.
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
    Attempt to evade a fingerprint hit by pitch shifting the entire signal. 
    Uses simple_fingerprinting: calculate BER between fingerprint of 
    original audio and of input, pitch shifted audio
    @param audio: Original audio digital signal
    @param sr: Audio's sample rate
    @param n_steps: Number of semitones to raise the audio's pitch
    @param write: Whether to write to a file or not

    Returns:
    Bit error rate (0 <= BER <= 1) of entire signal between 
    subfingerprints of the original audio and the shifted audio.
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
    Attempt to evade a fingerprint hit by pitch shifting the entire signal. 
    Uses pitch-robust fingerprinting scheme.
    This fingerprinting scheme consist of computing pitch-robust fingerprints, which 
    includes the fingerprint of the original signal as well as fingerprints of 
    pitch shifted (by different, specified amounts) versions of the signal.
    See get_robust_fps for the details on how much is the signal is
    pitch shifted by for each fingerprint
    
    @param audioname: Do not include file extension
    @param n_sts: Number of semitones to pitch shift the input audio by
    @param write: Whether to write the pitch-shifted audio to a wav file

    Returns: A BER dictionary: each entry is the Bit Error Rate between
    the our pitch_shifted input audio's fingerprint and one of the 
    fingerprints in the pitch-robust fingerprint dictionary, indexed by the amount
    of pitch shifting applied to the robust fingerprint
    
    """
    x, sr = librosa.load(f'{AUDIO_DIR}/{audioname}/{audioname}.wav')
    x_shift = pyrb.pitch_shift(x, sr, n_sts)
    if write:
        sf.write(f'{AUDIO_DIR}/{audioname}/{audioname}{float_filename(n_sts)}.wav', x_shift, sr)

    x_shift_sfps = get_fp(get_mel_spec(x_shift, sr))

    fp_list = get_robust_fps(x, sr, ST_OFFSET, STEP)
    BERs = {}
    for st in fp_list.keys():
        BER = round(get_BER(x_shift_sfps, fp_list[st]), 3)
        BERs[st] = BER
    return BERs


