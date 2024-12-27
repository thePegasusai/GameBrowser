// generate.js

import * as tf from '@tensorflow/tfjs';
import { DiT } from './dit.js';
import { AutoencoderKL } from './vae.js';
import { sigmoidBetaSchedule, createActionEmbedding } from './utils.js';

async function generateVideo(
    videoTensor,
    actions,
    ditCheckpoint = null,
    vaeCheckpoint = null,
    totalFrames = 32,
    maxNoiseLevel = 1000,
    ddimNoiseSteps = 100,
    noiseAbsMax = 20,
    ctxMaxNoiseIdx = null,
    nPromptFrames = 1
    ) {

    const model = new DiT();
    if(ditCheckpoint) {
      model.loadWeights(ditCheckpoint)
    }
    const vae = new AutoencoderKL();
    if(vaeCheckpoint) {
        vae.loadWeights(vaeCheckpoint)
    }

    const B = 1; // batch size

    // sampling params
   if(!ctxMaxNoiseIdx) {
      ctxMaxNoiseIdx = Math.floor(ddimNoiseSteps / 10 * 3);
    }
    const noiseRange = tf.linspace(-1, maxNoiseLevel - 1, ddimNoiseSteps + 1);

    // vae encoding
    const scalingFactor = 0.07843137255;

    let x = videoTensor.slice([0, 0], [-1, nPromptFrames]);
    x = tf.reshape(x, [B * nPromptFrames, x.shape[2], x.shape[3], x.shape[4]]);

    const H = x.shape[2];
    const W = x.shape[3];

    let encodedX;
    tf.tidy(() => {
        encodedX = vae.encode(tf.sub(tf.mul(x, 2), 1)).mean.mul(scalingFactor);
    });
    encodedX = tf.reshape(encodedX, [B, nPromptFrames, encodedX.shape[1], H / vae.patchSize, W / vae.patchSize]);

     // get alphas
    const betas = sigmoidBetaSchedule(maxNoiseLevel);
    const alphas = tf.sub(1, betas);
    const alphasCumprod = tf.cumprod(alphas, 0);
     const alphasCumprodReshaped = tf.reshape(alphasCumprod, [-1, 1, 1, 1]);
     
    let embeddedActions;
    tf.tidy(() => {
        embeddedActions = createActionEmbedding(actions);
    })

    // sampling loop
     let generatedFrames = encodedX.clone();
    for (let i = nPromptFrames; i < totalFrames; i++) {
         let chunk;
         tf.tidy(()=>{
           chunk = tf.randomNormal([B, 1, ...encodedX.shape.slice(-3)])
           chunk = tf.clipByValue(chunk, -noiseAbsMax, noiseAbsMax)
          });

        generatedFrames = tf.concat([generatedFrames, chunk], 1);
        const startFrame = Math.max(0, i + 1 - model.maxFrames);
         for (let noiseIdx = ddimNoiseSteps; noiseIdx >= 1; noiseIdx--) {

            const ctxNoiseIdx = Math.min(noiseIdx, ctxMaxNoiseIdx);
            let tCtx = tf.fill([B, i], noiseRange.slice([ctxNoiseIdx], [1]).arraySync()[0], 'int32')
             let t = tf.fill([B, 1], noiseRange.slice([noiseIdx], [1]).arraySync()[0], 'int32');
             let tNext = tf.fill([B, 1], noiseRange.slice([noiseIdx - 1], [1]).arraySync()[0], 'int32');
             tNext = tf.where(tf.less(tNext, 0), t, tNext);
             t = tf.concat([tCtx, t], 1);
             tNext = tf.concat([tCtx, tNext], 1)

            let xCurr = generatedFrames.clone();
             xCurr = xCurr.slice([0, startFrame]);
             t = t.slice([0], [-1], [startFrame]);
              tNext = tNext.slice([0], [-1], [startFrame]);


             let ctxNoise;
             tf.tidy(()=> {
               ctxNoise = tf.randomNormal(xCurr.slice([0,0],[-1, -1]).shape)
              ctxNoise = tf.clipByValue(ctxNoise, -noiseAbsMax, noiseAbsMax);
              xCurr = tf.add(tf.mul(tf.sqrt(alphasCumprodReshaped.slice([t.slice([0,0], [-1, -1]).arraySync()[0]])), xCurr.slice([0,0],[-1, -1])), tf.mul(tf.sqrt(tf.sub(1, alphasCumprodReshaped.slice([t.slice([0,0], [-1, -1]).arraySync()[0]]))), ctxNoise)

            });

             let v;
            tf.tidy(() => {
                v = model.apply(xCurr, tf.cast(t, 'int32'), embeddedActions.slice([0],[-1], [startFrame], [i + 1]))
            });
              let xStart;
              tf.tidy(() => {
                   xStart = tf.sub(tf.mul(tf.sqrt(alphasCumprodReshaped.slice([t.arraySync()[0]])), xCurr), tf.mul(tf.sqrt(tf.sub(1, alphasCumprodReshaped.slice([t.arraySync()[0]]))), v));
             });
              let xNoise;
              tf.tidy(() => {
                xNoise = tf.div(tf.sub(tf.mul(tf.sqrt(tf.div(1,alphasCumprodReshaped.slice([t.arraySync()[0]]))),xCurr), xStart), tf.sqrt(tf.sub(tf.div(1,alphasCumprodReshaped.slice([t.arraySync()[0]])), 1)));
              });


            let xPred;
              tf.tidy(() => {
               xPred =  tf.add(tf.mul(tf.sqrt(alphasCumprodReshaped.slice([tNext.arraySync()[0]])), xStart), tf.mul(xNoise, tf.sqrt(tf.sub(1, alphasCumprodReshaped.slice([tNext.arraySync()[0]])))))
            })
             generatedFrames = generatedFrames.slice([0],[-1], [0], [generatedFrames.shape[1]-1]).concat(xPred.slice([0], [-1], [-1], [-1], [-1]), 1);
        }
     }


    // vae decoding
     let decodedFrames;
     tf.tidy(() => {
        decodedFrames = tf.reshape(generatedFrames, [B * totalFrames, generatedFrames.shape[2], generatedFrames.shape[3] * generatedFrames.shape[4], generatedFrames.shape[5]])
        decodedFrames = tf.div(tf.add(vae.decode(tf.div(decodedFrames, scalingFactor)), 1), 2);
         decodedFrames = tf.reshape(decodedFrames, [B, totalFrames, decodedFrames.shape[1], H, W, 3]);
     })

      return decodedFrames;
}


export { generateVideo };