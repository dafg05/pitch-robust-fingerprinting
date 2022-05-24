import numpy as np
import librosa
import librosa.display
import pyrubberband as pyrb
import math
from bitstring import BitArray

# each frame should be around 0.37 seconds
FRAME_DURATION = 0.37
# overlap factor: 31/32
HOP_DURATION = 0.037/32
NUM_OF_BANDS = 33
FMIN = 300
FMAX = 2000
FB_BLOCK_LEN = 256

def get_mel_spec(x: np.ndarray, sr: int) -> np.ndarray:
    """
    Generates an energy mel-scaled-spectrogram with NUM_OF_BANDS frequency bins 
    between FMIN and FMAX
    @param x: Audio digital signal
    @param sr: Audio's sample rate

    Returns:
    mel-scaled-spectrogram
    """
    n_fft = math.ceil(FRAME_DURATION * sr)
    hop_length = math.ceil(HOP_DURATION * sr)
    return librosa.feature.melspectrogram(x, sr=sr, n_fft=n_fft, hop_length=hop_length, power=1, n_mels = NUM_OF_BANDS, fmin = FMIN, fmax = FMAX)

def sfp_bit(energy_spec: np.ndarray, frame_num: int, band_num: int) -> bool:
    """
    Calculates the band_numth bit of the sub-fingerprint
    that corresponds to the frame_numth frame.
    Note the edge case: if we're trying to calculate the bit for the 0th frame,
    then frame -1 is defined as the last frame of the spectrogram

    @param energy_spec: mel-scaled spectrogram, specifics in get_mel_spec()
    @param frame_num: index of frame of current subfingerprint
    @param band_num: index of frequency band of current bit

    Returns:
    Bit value (bool)
    """
    diff_current_frame = energy_spec[band_num][frame_num] - energy_spec[band_num+1][frame_num]
    diff_prev_frame = energy_spec[band_num][frame_num-1] - energy_spec[band_num+1][frame_num-1]
    return (diff_current_frame - diff_prev_frame) > 0

def get_sfp(energy_spec: np.ndarray, frame_num: int) -> BitArray:
    """
    Generates a 32 bit fingerprint for the frame_numth frame.

    @param energy_spec: mel-scaled spectrogram, specifics in get_mel_spec()
    @param frame_num: index of frame of current subfingerprint

    Returns:
    BitArray of 32 bits.
    """
    sfp = BitArray('')
    for i in range(32):
        if sfp_bit(energy_spec, frame_num = frame_num, band_num = i):
            sfp.append('0b1')
        else:
            sfp.append('0b0')
    assert len(sfp) == 32, "Something went wrong"
    return sfp

def extract_all_sfps(energy_spec: np.ndarray) -> list:
    """
    Calculates subfingerprints from an audio signal
    given its mel-scaled-spectrogram.

    @param energy_spec  mel-scaled spectrogram, specifics in get_mel_spec()

    Returns:
    List of subfingerprints corresponding to the index of each audio frame
    """
    assert energy_spec.shape[0] == NUM_OF_BANDS, "Spectrogram must have NUM_OF_BANDS frequency bins"
    sfps = []
    for i in range(energy_spec.shape[1]):
        sfps.append(get_sfp(energy_spec, frame_num = i))
    return sfps

def compute_fp_block(sfps: list, starting_frame: int):
    """
    Extracts a fingeprint block, a sequence of 256 subfingerprints, starting
    from the starting_framenth subfingerprint of an audio signal.
    """
    assert len(sfps) > starting_frame + FB_BLOCK_LEN-1 , "Not enough subfingerprints to compute fp_block"
    fp_block = []
    for i in range(starting_frame, starting_frame + FB_BLOCK_LEN):
        fp_block.append(sfps[starting_frame + i])
    
    assert len(fp_block) == FB_BLOCK_LEN, "Something went wrong"
    return fp_block

def hamming_distance(a: BitArray, b: BitArray) -> int:
    """
    From: https://bitstring.readthedocs.io/en/latest/walkthrough.html
    Return the number of bit errors between two equally sized BitArrays
    """
    return (a^b).count(True)

def get_BER(sfps1: np.ndarray, sfps2: np.ndarray) -> float:
    """
    Returns the bit error rate between two list of subfingerprints
    """
    assert len(sfps1) == len(sfps2), "Not the same number of subfingerprints"
    total_bits = (NUM_OF_BANDS-1) * len(sfps1)
    bit_errors = 0
    for i in range(len(sfps1)):
        bit_errors += hamming_distance(sfps1[i], sfps2[i])
    return bit_errors/total_bits
        
def compare_whole_signal(energy_spec1: np.ndarray, energy_spec2: np.ndarray) -> float:
    """
    Returns the bit error rate of the list of subfingerprints 
    between two audio signals
    """
    assert energy_spec1.shape == energy_spec2.shape, "Both spectrograms must have equal shape"
    assert energy_spec1.shape[0] == NUM_OF_BANDS, "Both spectrograms must have 33 frequency bins"
    sfps1 = extract_all_sfps(energy_spec1)
    sfps2 = extract_all_sfps(energy_spec2)
    return get_BER(sfps1, sfps2)

def extract_shifted_fps(x: np.ndarray, sr: int, st_offset: float, step: float) -> dict:
    """
    Extract a robust (to pitch shifitng) set of fingerprints.
    @param x: digital audio signal
    @param sr: sample rate
    @param st_offset: positive number, max number of 
    semitones to shift x by (from zero).

    Example: If offset = 1, then the range of semitones
    we'll pitch shift x by will be [-1, 1].

    @param step: step size between pitch transformations.
    
    Example: if step = 0.5 and st_offset = 1, then we'll pitch shift 
    the original signal by -1, -0.5, 0.5, and 1 semitones.
    Example: if step = 0.4 and st_offset = 1, then we'll pitch shift
    the original signal by -1, -0.6,-0.2, 0.2, 0.6, and 1 semitones

    Returns:
    Robust fingerprint dictionary: each entry is a fingerprint indexed
    by the amount of pitch shifting applied to compute said fingerprint
    """

    assert st_offset > 0, "st_offset must be a positive number"
    assert step > 0, "step must be a positive number"
    assert step <= st_offset, "step must be less than or equal to than st_offset"
    
    n = -st_offset # variable that holds number of semitones to pitch shift by
    shift_list = [] # list containing all instances of n
    while(n <= st_offset):
        shift_list.append(n)
        n += step

    fp_dict = {}
    for i in shift_list:
        if i == 0:
            fp_dict[i] = extract_all_sfps(get_mel_spec(x, sr))
        else:
            x_shift = pyrb.pitch_shift(x, sr, i)
            fp_dict[i] = extract_all_sfps(get_mel_spec(x_shift, sr))

    return fp_dict
        
        
    