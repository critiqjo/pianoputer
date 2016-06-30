#!/usr/bin/env python2

from scipy.io import wavfile
import argparse
import numpy as np
import pygame
import sys
import warnings


def speedx(snd_array, factor):
    """ Speeds up / slows down a sound, by some factor. """
    indices = np.round(np.arange(0, len(snd_array), factor))
    indices = indices[indices < len(snd_array)].astype(int)
    return snd_array[indices]


def stretch(snd_array, factor, window_size, h):
    """ Stretches/shortens a sound, by some factor. """
    phase = np.zeros(window_size)
    hanning_window = np.hanning(window_size)
    result = np.zeros(len(snd_array) / factor + window_size)

    for i in np.arange(0, len(snd_array) - (window_size + h), h*factor):
        # Two potentially overlapping subarrays
        a1 = snd_array[i: i + window_size]
        a2 = snd_array[i + h: i + window_size + h]

        # The spectra of these arrays
        s1 = np.fft.fft(hanning_window * a1)
        s2 = np.fft.fft(hanning_window * a2)

        # Rephase all frequencies
        phase = (phase + np.angle(s2/s1)) % 2*np.pi

        a2_rephased = np.fft.ifft(np.abs(s2)*np.exp(1j*phase))
        i2 = int(i/factor)
        result[i2: i2 + window_size] += hanning_window*a2_rephased.real

    # normalize (16bit)
    result = ((2**(16-4)) * result/result.max())

    return result.astype('int16')


def pitchshift(snd_array, n, window_size=2**13, h=2**11):
    """ Changes the pitch of a sound by ``n`` semitones. """
    factor = 2**(1.0 * n / 12.0)
    stretched = stretch(snd_array, 1.0/factor, window_size, h)
    return speedx(stretched[window_size:], factor)


def parse_arguments():
    description = ('Use your computer keyboard as a "piano"')

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--wav', '-w',
        metavar='FILE',
        type=argparse.FileType('r'),
        default='bowl.wav',
        help='WAV file (default: bowl.wav)')
    parser.add_argument(
        '--keyboard', '-k',
        metavar='FILE',
        type=argparse.FileType('r'),
        default='typewriter.kb',
        help='keyboard file (default: typewriter.kb)')
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='verbose mode')

    return (parser.parse_args(), parser)


def main():
    # Parse command line arguments
    (args, parser) = parse_arguments()

    # Enable warnings from scipy if requested
    if not args.verbose:
        warnings.simplefilter('ignore')

    fps, sound = wavfile.read(args.wav.name)
    keys = [k for k in args.keyboard.read().split('\n') if k]
    if len(keys) != 36:
        raise Exception("Number of keys must be exactly 36!")
    key_idx_lookup = dict(zip(keys, range(0, 36)))
    if len(key_idx_lookup) != 36:
        raise Exception("There are duplicates in keyboard file!")

    tones = range(-30, 31)
    sys.stdout.write('Transponding sound file... ')
    sys.stdout.flush()
    transposed_sounds = [pitchshift(sound, n) for n in tones]
    print('DONE')

    # So flexible ;)
    pygame.mixer.init(fps, -16, 1, 2048)
    # For the focus
    screen = pygame.display.set_mode((150, 150))

    sounds = map(pygame.sndarray.make_sound, transposed_sounds)
    is_playing = [False for n in tones]
#    key_sound = dict(zip(keys, sounds))
#    is_playing = {k: False for k in keys}

    while True:
        event = pygame.event.wait()

        if event.type in (pygame.KEYDOWN, pygame.KEYUP):
            key = pygame.key.name(event.key)
            if key not in key_idx_lookup:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    raise KeyboardInterrupt
                else:
                    continue
            k_idx = key_idx_lookup[key]
            k_grp = k_idx / 7
            k_off = k_idx % 7
            s_idx = 12 * k_grp
            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                if k_grp == 5 or k_off in [2, 6]:
                    continue
                elif k_off < 2:
                    s_idx += 2 * k_off + 1
                else:
                    s_idx += 2 * k_off
            else:
                if k_off < 3:
                    s_idx += 2 * k_off
                else:
                    s_idx += 2 * k_off - 1

        if event.type == pygame.KEYDOWN:
            print(s_idx)
            if not is_playing[s_idx]:
                sounds[s_idx].play(fade_ms=50)
                is_playing[s_idx] = True

        elif event.type == pygame.KEYUP:
            sounds[s_idx].fadeout(400)
            is_playing[s_idx] = False


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Goodbye')
