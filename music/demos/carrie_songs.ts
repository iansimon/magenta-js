/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import * as tf from '@tensorflow/tfjs';
import * as mm from '../src/index';

import {CHECKPOINTS_DIR} from './common';
import {writeNoteSeqs} from './common';

const MEL_MULTI_DIR = `${CHECKPOINTS_DIR}/music_vae/mel_multicontrol`;
const MEL_MULTI_CONTROLS_KEY =
    `${MEL_MULTI_DIR}/mel_2bar_multicontrol_key_tiny_fb16`;

interface Song {
  id: string;
  melody: number[][];
  chords: string[][];
}
interface Songs {
  [name: string]: Song;
}
const SONGS: Songs = {
  'Hey Jude': {
    'id': 'heyjude',
    'melody': [
      [
        60, -2, -2, -2, -2, -2, -2, -2, 64, -2, -2, -2, 67, -2, -2, -2,
        74, 72, 74, -2, 72, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2
      ],
      [
        74, 72, 74, -2, 72, -2, -2, -2, -2, -2, -2, -2, 70, -2, 69, -2,
        67, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2
      ],
      [
        60, -2, -2, -2, -2, -2, -2, -2, 64, -2, -2, -2, 67, -2, -2, -2,
        74, 72, 74, -2, 72, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2
      ],
      [
        74, 72, 74, -2, 72, -2, -2, -2, -2, -2, -2, -2, 70, -2, 69, -2,
        67, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2
      ]
    ],
    'chords': [['C', 'Bb'], ['F', 'C'], ['C', 'Bb'], ['F', 'C']]
  },

  'Somewhere Over the Rainbow': {
    'id': 'rainbow',
    'melody': [
      [
        60, -2, -2, -2, -2, -2, -2, -2, 72, -2, -2, -2, -2, -2, -2, -2,
        71, -2, -2, -2, 67, -2, 69, -2, 71, -2, -2, -2, 72, -2, -2, -2
      ],
      [
        60, -2, -2, -2, -2, -2, -2, -2, 69, -2, -2, -2, -2, -2, -2, -2,
        67, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2
      ],
      [
        57, -2, -2, -2, -2, -2, -2, -2, 65, -2, -2, -2, -2, -2, -2, -2,
        64, -2, -2, -2, 60, -2, 62, -2, 64, -2, -2, -2, 65, -2, -2, -2
      ],
      [
        62, -2, -2, -2, 59, -2, 60, -2, 62, -2, -2, -2, 64, -2, -2, -2,
        60, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2
      ]
    ],
    'chords': [['C', 'Em'], ['F', 'C'], ['F', 'C'], ['G', 'C']]
  },

  'Pumped Up Kicks': {
    'id': 'kicks',
    'melody': [
      [
        76, -2, 74, -2, 74, -2, 72, -2, 72, -2, -1, -2, 69, -2, 67, -2,
        72, -2, -1, -2, 72, -2, -1, -2, 72, -2, 67, -2, 69, -2, 67, -2
      ],
      [
        72, -2, -1, -2, 72, -2, 72, -2, 72, -2, -1, -2, -2, -2, 74, -2,
        -1, -2, 74, -2, -1, -2, 76, -2, 74, -2, -1, -2, -2, -2, -2, -2
      ],
      [
        76, -2, 74, -2, 74, -2, 72, -2, 72, -2, -1, -2, 69, -2, 67, -2,
        72, -2, -1, -2, 72, -2, -1, -2, 72, -2, 67, -2, 69, -2, 67, -2
      ],
      [
        72, -2, -1, -2, 72, -2, 72, -2, 72, -2, -1, -2, -2, -2, 74, -2,
        -1, -2, 74, -2, 74, -2, 76, -2, 74, -2, 72, -2, -1, -2, -2, -2
      ]
    ],
    'chords': [['Dm', 'F'], ['C', 'G'], ['Dm', 'F'], ['C', 'G']]
  },

  'I Want to Hold Your Hand': {
    'id': 'holdhand',
    'melody': [
      [
        67, -2, -2, 65, 64, -2, -2, -2, -2, -2, -2, -2, 64, -2, 67, -2,
        65, -2, -2, 64, 62, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2
      ],
      [
        -1, -2, 64, -2, 64, -2, 64, -2, 64, -2, -2, -2, 64, -2, -2, -2,
        59, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 69, -2, -2, -2
      ],
      [
        67, -2, -2, 65, 64, -2, -2, -2, -2, -2, -2, -2, 64, -2, 67, -2,
        65, -2, -2, 64, 62, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2
      ],
      [
        -1, -2, 64, -2, 64, -2, 64, -2, 64, -2, -2, -2, 64, -2, -2, -2,
        76, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2
      ]
    ],
    'chords': [['C', 'G'], ['Am', 'Em'], ['C', 'G'], ['Am', 'E']]
  },

  'Bad Romance': {
    'id': 'romance',
    'melody': [
      [
        65, -2, -2, -2, -2, -2, 64, -2, 65, -2, 64, -2, -2, -2, 62, -2,
        -2, -2, -2, -2, -2, -2, 59, -2, 60, -2, -2, -2, 62, -2, -2, -2
      ],
      [
        64, -2, -2, -2, 64, -2, 64, -2, 64, -2, 62, -2, -2, -2, 60, -2,
        -2, -2, -2, -2, -2, -2, -2, -2, 60, -2, 62, -2, 64, -2, 60, -2
      ],
      [
        65, -2, -2, -2, -2, -2, 64, -2, 65, -2, 64, -2, -2, -2, 62, -2,
        -2, -2, -2, -2, -2, -2, 59, -2, 60, -2, -2, -2, 62, -2, -2, -2
      ],
      [
        64, -2, -2, -2, 64, -2, 64, -2, 64, -2, 62, -2, -2, -2, 60, -2,
        -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2
      ],
    ],
    'chords': [['F', 'G'], ['Am', 'C'], ['F', 'G'], ['E', 'Am']]
  }
};

function melodyToNoteSequence(melody: number[]) {
  return (new mm.melodies.Melody(melody.map(e => e + 2), 0, 127))
      .toNoteSequence();
}

const RHYTHM = new mm.melodies.MelodyRhythm();
const SHAPE = new mm.melodies.MelodyShape();
const REGISTER = new mm.melodies.MelodyRegister([50, 63, 70]);

async function runSong(
    id: string, mvae: mm.MusicVAE, melodies: number[][], chords: string[][]) {
  const melSeqs = melodies.map(mel => melodyToNoteSequence(mel));
  const mels = melSeqs.map(
      seq => mm.melodies.Melody.fromNoteSequence(seq, 0, 127, undefined, 32));
  writeNoteSeqs(`mel-${id}-orig`, melSeqs);

  const melRhythms = mels.map((mel) => RHYTHM.extract(mel));
  const melShapes = mels.map((mel) => SHAPE.extract(mel));
  const melRegisters = mels.map((mel) => REGISTER.extract(mel));

  const z: tf.Tensor2D = tf.randomNormal([1, mvae.zDims]);

  const promises = [0, 1, 2, 3].map(i => mvae.decode(z, null, {
    chordProgression: chords[i],
    key: 0,
    extraControls:
        {rhythm: melRhythms[i], shape: melShapes[i], register: melRegisters[0]}
  }));
  const samples = (await Promise.all(promises)).map(s => s[0]);

  writeNoteSeqs(`mel-${id}-sample`, samples);

  for (const i of [0, 1, 2, 3]) {
    melRhythms[i].dispose();
    melShapes[i].dispose();
    melRegisters[i].dispose();
  }

  z.dispose();
}

async function generateSongButton(name: string, mvae: mm.MusicVAE) {
  const songButton = document.createElement('button');
  songButton.textContent = 'Generate';
  songButton.addEventListener('click', () => {
    runSong(SONGS[name].id, mvae, SONGS[name].melody, SONGS[name].chords);
    songButton.disabled = true;
  });
  const songDiv = document.getElementById(`generate-${SONGS[name].id}`);
  songDiv.appendChild(songButton);
}

try {
  const mvae = new mm.MusicVAE(MEL_MULTI_CONTROLS_KEY);

  mvae.initialize().then(() => {
    Promise.all(
        [Object.keys(SONGS).map((name) => generateSongButton(name, mvae))]);
  });
} catch (err) {
  console.error(err);
}
