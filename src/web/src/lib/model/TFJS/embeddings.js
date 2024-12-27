// embeddings.js

import * as tf from '@tensorflow/tfjs';

class Timesteps {
    constructor(numChannels, flipSinToCos = true, downscaleFreqShift = 0) {
        this.numChannels = numChannels;
        this.flipSinToCos = flipSinToCos;
        this.downscaleFreqShift = downscaleFreqShift;
    }

    forward(timesteps) {
        return this.getTimestepEmbedding(timesteps, this.numChannels, this.flipSinToCos, this.downscaleFreqShift);
    }

     getTimestepEmbedding(timesteps, embeddingDim, flipSinToCos = false, downscaleFreqShift = 0, scale = 1, maxPeriod = 10000) {
        if (timesteps.shape.length !== 1 && timesteps.shape.length !== 2) {
            throw new Error("Timesteps should be a 1D or 2D tensor");
        }

        const halfDim = Math.floor(embeddingDim / 2);
         const exponent = tf.range(0, halfDim).toFloat().mul(-Math.log(maxPeriod)).div(halfDim - downscaleFreqShift)

        let emb = tf.exp(exponent);
        emb = timesteps.reshape([...timesteps.shape, 1]).toFloat().mul(emb);

        emb = emb.mul(scale);

        const sinEmb = tf.sin(emb);
        const cosEmb = tf.cos(emb);

        let embCombined = tf.concat([sinEmb, cosEmb], -1);

        if (flipSinToCos) {
          const [cosPart, sinPart] = tf.split(embCombined, 2, -1);
          embCombined = tf.concat([cosPart, sinPart], -1);
        }

        if(embeddingDim % 2 == 1){
           embCombined = tf.pad(embCombined, [[0,0], [0, 1]]);
        }
        return embCombined;
    }
}


class Positions2d {
    constructor(numChannels, flipSinToCos = true, downscaleFreqShift = 0) {
        this.numChannels = numChannels;
        this.flipSinToCos = flipSinToCos;
        this.downscaleFreqShift = downscaleFreqShift;
    }

    forward(grid) {
      const hEmb = this.getTimestepEmbedding(grid[0], this.numChannels / 2, this.flipSinToCos, this.downscaleFreqShift);
      const wEmb = this.getTimestepEmbedding(grid[1], this.numChannels / 2, this.flipSinToCos, this.downscaleFreqShift);
      return tf.concat([hEmb, wEmb], -1);
    }

    getTimestepEmbedding(timesteps, embeddingDim, flipSinToCos = false, downscaleFreqShift = 0, scale = 1, maxPeriod = 10000) {
        if (timesteps.shape.length !== 1 && timesteps.shape.length !== 2) {
            throw new Error("Timesteps should be a 1D or 2D tensor");
        }

        const halfDim = Math.floor(embeddingDim / 2);
        const exponent = tf.range(0, halfDim).toFloat().mul(-Math.log(maxPeriod)).div(halfDim - downscaleFreqShift)

        let emb = tf.exp(exponent);
        emb = timesteps.reshape([...timesteps.shape, 1]).toFloat().mul(emb);

        emb = emb.mul(scale);

        const sinEmb = tf.sin(emb);
        const cosEmb = tf.cos(emb);

        let embCombined = tf.concat([sinEmb, cosEmb], -1);

        if (flipSinToCos) {
            const [cosPart, sinPart] = tf.split(embCombined, 2, -1);
            embCombined = tf.concat([cosPart, sinPart], -1);
        }

        if(embeddingDim % 2 == 1){
            embCombined = tf.pad(embCombined, [[0,0], [0, 1]]);
        }
        return embCombined;
    }
}

export { Timesteps, Positions2d };