# An audio fingerprinting scheme robust to pitch shifting

Required packages:

    pip install librosa
    pip install numpy
    pip install SoundFile
    pip install pyrubberband
    pip install bitstring

Original fingerprinting scheme proposed by Haitsma, et. al..
Link to paper:
http://ismir2002.ircam.fr/proceedings/02-FP04-2.pdf

**See final_paper.pdf for description of pitch-robust fingerprinting scheme.**

To run a number of simulations of a pitch attack on either the original or pitch-robust fingerprinting scheme, run:

    python3 test_attack.py <audioname> <defensetype>

Example: run pitch attacks on pitch-robust fingerprinting scheme using `brahms.wav` as the target audio:

    python3 test_attack.py brahms robust
