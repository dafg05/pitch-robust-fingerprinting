from attacks import pitch_attack_robust, pitch_attack_simple, fingerprint_hit, least_BER
import random
import sys

MAX_ST_OFFSET = 2
VALID_AUDIO_NAMES = ['brahms', 'choice', 'fishin', 'trumpet']
NUM_OF_TRIALS = 10
DEFENSE_TYPES  = ['simple', 'robust']

def test_pitch_attack_robust(audioname: str, n: int):
    """
    Try evading the pitch-robust fingerprinting scheme by pitch shifting
    an audio file.
    """
    
    print("I AM IN ROBUST")
    f = f = open(f"experiments/robust_results_{audioname}.txt", "w")
    f.write("Testing a pitch shifting attack against pitch robust fingerprinting scheme\n")
    f.write("--------------------------------------------------------------------------\n")

    for i in range(n):
        # -MAX_ST_OFFSET <= n_sts < MAX_ST_OFFSET, where n_sts is randomly chosen.
        n_sts =  ((MAX_ST_OFFSET * 2) * random.random()) - MAX_ST_OFFSET
        BERs = pitch_attack_robust(audioname, n_sts, False)

        # Write BERs on file
        f = open(f"experiments/robust_results_{audioname}.txt", "a")
        f.write(f"Attack on {audioname} with n_sts {n_sts} \n")
        f.write(f"BERs: {BERs} \n")

        if fingerprint_hit(least_BER(BERs)):
            f.write("FINGERPRINT HIT. ATTACK UNSUCCESSFUL\n")
        else:
            f.write("FINGERPRINT_MISS. ATTACK SUCCESSFUL\n")
        f.close

    print(f"Output written on: experiments/robust_results_{audioname}.txt")

def test_pitch_attack_simple(audioname: str, n: int):
    """
    Weh
    """

    f = f = open(f"experiments/simple_results_{audioname}.txt", "w")
    f.write("Testing a pitch shifting attack against simple fingerprinting scheme\n")
    f.write("--------------------------------------------------------------------------\n")
    for i in range(n):
        # -MAX_ST_OFFSET <= n_sts < MAX_ST_OFFSET, where n_sts is randomly chosen.
        n_sts =  ((MAX_ST_OFFSET * 2) * random.random()) - MAX_ST_OFFSET
        BER = pitch_attack_simple(audioname, n_sts, False)

        # Write BERs on file
        f = open(f"experiments/simple_results_{audioname}.txt", "a")
        f.write(f"Attack on {audioname} with n_sts {n_sts} \n")
        f.write(f"BER: {BER} \n")

        if fingerprint_hit(BER):
            f.write("FINGERPRINT HIT. ATTACK UNSUCCESSFUL\n")
        else:
            f.write("FINGERPRINT_MISS. ATTACK SUCCESSFUL\n")
        f.close

    print(f"Output written on: experiments/simple_results_{audioname}.txt")
        
exit = False

if len(sys.argv) != 3:
    print("USAGE: test_attack.py <audioname> <defense type>")
    sys.exit()

if not (sys.argv[1] in VALID_AUDIO_NAMES):
    print(f"Options for audioname are: {VALID_AUDIO_NAMES}. ")
    exit = True

if not (sys.argv[2] in DEFENSE_TYPES):
    print(f"Options for defense types are: {DEFENSE_TYPES}.")
    exit = True

if exit:
    sys.exit()

audioname = sys.argv[1]

globals()[f"test_pitch_attack_{sys.argv[2]}"](audioname, NUM_OF_TRIALS)