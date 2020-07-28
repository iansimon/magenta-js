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

import {performance} from '../src/core/compat/global';
import * as mm from '../src/index';

import {CHECKPOINTS_DIR, DRUM_SEQS, MEL_TWINKLE} from './common';
import {writeMemory, writeNoteSeqs, writeTimer} from './common';

mm.logging.verbosity = mm.logging.Level.DEBUG;

const TRIO_INFILL_CKPT = `${CHECKPOINTS_DIR}/music_vae/trio_2bar_infill`;

export const INITIAL_TRIO: mm.INoteSequence = {
  notes: [],
  quantizationInfo: {stepsPerQuarter: 4},
  totalQuantizedSteps: 32
};

MEL_TWINKLE.notes.map(n => {
  const m = mm.NoteSequence.Note.create(n);
  m.program = 0;
  m.instrument = 0;
  INITIAL_TRIO.notes.push(m);
});

MEL_TWINKLE.notes.map(n => {
  const m = mm.NoteSequence.Note.create(n);
  m.pitch -= 24;
  m.program = 32;
  m.instrument = 1;
  INITIAL_TRIO.notes.push(m);
});

DRUM_SEQS[0].notes.map(n => {
  const m = mm.NoteSequence.Note.create(n);
  m.instrument = 2;
  INITIAL_TRIO.notes.push(m);
});

async function runTrio() {
  const mvae = new mm.MusicVAE(TRIO_INFILL_CKPT);
  await mvae.initialize();

  writeNoteSeqs('trio-inputs', [INITIAL_TRIO], true);

  let start = performance.now();

  const trioTensor = mvae.dataConverter.toTensor(INITIAL_TRIO);
  const melTensor = trioTensor.slice([0, 0], [32, 90]);
  const bassTensor = trioTensor.slice([0, 90], [32, 90]);
  const drumsTensor = trioTensor.slice([0, 180], [32, 512]);

  const emptyMelTensor = tf.zerosLike(melTensor);
  const emptyBassTensor = tf.zerosLike(bassTensor);
  const emptyDrumsTensor = tf.zerosLike(drumsTensor);

  const melInfillTensor =
      tf.concat2d([tf.ones([32, 1]), tf.zeros([32, 1]), tf.zeros([32, 1])], 1);
  const bassInfillTensor =
      tf.concat2d([tf.zeros([32, 1]), tf.ones([32, 1]), tf.zeros([32, 1])], 1);
  const drumsInfillTensor =
      tf.concat2d([tf.zeros([32, 1]), tf.zeros([32, 1]), tf.ones([32, 1])], 1);

  const newMelSeqs = await mvae.sample(1, null, {
    extraControls: {
      melody: emptyMelTensor,
      bass: bassTensor,
      drums: drumsTensor,
      infill: melInfillTensor,
    }
  });

  const newBassSeqs = await mvae.sample(1, null, {
    extraControls: {
      melody: melTensor,
      bass: emptyBassTensor,
      drums: drumsTensor,
      infill: bassInfillTensor,
    }
  });

  const newDrumsSeqs = await mvae.sample(1, null, {
    extraControls: {
      melody: melTensor,
      bass: bassTensor,
      drums: emptyDrumsTensor,
      infill: drumsInfillTensor,
    }
  });

  writeTimer('trio-sample-time', start);
  writeNoteSeqs(
      'trio-samples', [newMelSeqs[0], newBassSeqs[0], newDrumsSeqs[0]], true);

  start = performance.now();

  const similarMelSeqs = await mvae.similar(INITIAL_TRIO, 1, 0.9, null, {
    extraControls: {
      melody: emptyMelTensor,
      bass: bassTensor,
      drums: drumsTensor,
      infill: melInfillTensor,
    }
  });

  const similarBassSeqs = await mvae.similar(INITIAL_TRIO, 1, 0.9, null, {
    extraControls: {
      melody: melTensor,
      bass: emptyBassTensor,
      drums: drumsTensor,
      infill: bassInfillTensor,
    }
  });

  const similarDrumsSeqs = await mvae.similar(INITIAL_TRIO, 1, 0.9, null, {
    extraControls: {
      melody: melTensor,
      bass: bassTensor,
      drums: emptyDrumsTensor,
      infill: drumsInfillTensor,
    }
  });

  writeTimer('trio-similar-time', start);
  writeNoteSeqs(
      'trio-similar',
      [similarMelSeqs[0], similarBassSeqs[0], similarDrumsSeqs[0]], true);

  trioTensor.dispose();
  melTensor.dispose();
  bassTensor.dispose();
  drumsTensor.dispose();

  emptyMelTensor.dispose();
  emptyBassTensor.dispose();
  emptyDrumsTensor.dispose();

  melInfillTensor.dispose();
  bassInfillTensor.dispose();
  drumsInfillTensor.dispose();

  mvae.dispose();
}

async function generateAllButton() {
  const button = document.createElement('button');
  button.textContent = 'Generate All';
  button.addEventListener('click', () => {
    runTrio();
    button.disabled = true;
  });
  const div = document.getElementById('generate-all');
  div.appendChild(button);
}

async function generateTrioButton() {
  const trioButton = document.createElement('button');
  trioButton.textContent = 'Generate Trio via Infilling';
  trioButton.addEventListener('click', () => {
    runTrio();
    trioButton.disabled = true;
  });
  const trioDiv = document.getElementById('generate-trio');
  trioDiv.appendChild(trioButton);
}

try {
  Promise
      .all([
        generateAllButton(),
        generateTrioButton(),
      ])
      .then(() => writeMemory(tf.memory().numBytes));
} catch (err) {
  console.error(err);
}
