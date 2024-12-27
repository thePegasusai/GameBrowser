// dit.js

import * as tf from '@tensorflow/tfjs';
import { Timesteps } from './embeddings.js';
import { SpatialAxialAttention, TemporalAxialAttention } from './attention.js';
import { RotaryEmbedding } from './rotary_embedding.js';

class PatchEmbed extends tf.layers.Layer {
    constructor(imgHeight, imgWidth, patchSize, inChans, embedDim, flatten = true) {
        super({});
        this.imgHeight = imgHeight;
        this.imgWidth = imgWidth;
        this.patchSize = patchSize;
        this.inChans = inChans;
        this.embedDim = embedDim;
        this.flatten = flatten;

        this.gridSize = [Math.floor(imgHeight / patchSize), Math.floor(imgWidth / patchSize)];
        this.numPatches = this.gridSize[0] * this.gridSize[1];

        this.proj = tf.layers.conv2d({
            filters: embedDim,
            kernelSize: patchSize,
            strides: patchSize,
            padding: 'valid',
            useBias: true,
            kernelInitializer: 'glorotUniform',
            biasInitializer: 'zeros'
        });
        this.norm = tf.layers.layerNormalization({epsilon: 1e-6});
    }

    call(x) {
        let B = x.shape[0];
        let C = x.shape[1];
        let H = x.shape[2];
        let W = x.shape[3];

        if(!(H == this.imgHeight && W == this.imgWidth)){
            throw new Error(`Input image size (${H}*${W}) doesn't match model (${this.imgHeight}*${this.imgWidth}).`)
        }
        
        x = this.proj.apply(x);

        if (this.flatten) {
            x = tf.reshape(x, [B, this.gridSize[0] * this.gridSize[1], this.embedDim]);
        }
        else {
            x = tf.reshape(x, [B, this.gridSize[0], this.gridSize[1], this.embedDim])
        }
        x = this.norm.apply(x);
        return x;
    }
}

class TimestepEmbedder extends tf.layers.Layer {
    constructor(hiddenSize, frequencyEmbeddingSize = 256) {
        super({});
        this.hiddenSize = hiddenSize;
        this.frequencyEmbeddingSize = frequencyEmbeddingSize;
         this.mlp = [
            tf.layers.dense({ units: hiddenSize, useBias: true, kernelInitializer: 'glorotUniform', biasInitializer: 'zeros'}),
            tf.layers.activation({activation: 'silu'}),
            tf.layers.dense({ units: hiddenSize, useBias: true, kernelInitializer: 'glorotUniform', biasInitializer: 'zeros'}),
         ]

    }


    call(t) {
      const tFreq = this.timestepEmbedding(t, this.frequencyEmbeddingSize);
      let tEmb = this.mlp[0].apply(tFreq);
      tEmb = this.mlp[1].apply(tEmb)
      tEmb = this.mlp[2].apply(tEmb);
      return tEmb;
    }


    timestepEmbedding(t, dim, maxPeriod = 10000) {
      const half = Math.floor(dim / 2);
      const freqs = tf.exp(tf.range(0, half).mul(-Math.log(maxPeriod)).div(half));
      const args = tf.mul(tf.reshape(t, [-1, 1]).toFloat(), freqs);
      const embedding = tf.concat([tf.cos(args), tf.sin(args)], -1);
      if (dim % 2) {
        embedding = tf.concat([embedding, tf.zerosLike(embedding.slice([0, 0], [-1, 1]))], -1);
      }
      return embedding;
    }
}


class FinalLayer extends tf.layers.Layer {
    constructor(hiddenSize, patchSize, outChannels) {
        super({});
        this.hiddenSize = hiddenSize;
        this.patchSize = patchSize;
        this.outChannels = outChannels;
        this.normFinal = tf.layers.layerNormalization({epsilon: 1e-6, trainable:false});
        this.linear = tf.layers.dense({units: patchSize * patchSize * outChannels, useBias: true, kernelInitializer: tf.initializers.zeros(), biasInitializer: tf.initializers.zeros()});
        this.adaLNModulation = [
            tf.layers.activation({activation: 'silu'}),
            tf.layers.dense({units: 2 * hiddenSize, useBias: true, kernelInitializer: tf.initializers.zeros(), biasInitializer: tf.initializers.zeros()})
        ]
    }

    call(x, c) {
        let modulation = this.adaLNModulation[0].apply(c);
        modulation = this.adaLNModulation[1].apply(modulation);
        const [shift, scale] = tf.split(modulation, 2, -1);
        x = this.modulate(this.normFinal.apply(x), shift, scale);
        x = this.linear.apply(x);
        return x;
    }

    modulate(x, shift, scale) {
      let fixedDims = Array(shift.shape.length - 1).fill(1)
      let shiftRepeated = tf.broadcastTo(shift, [x.shape[0], ...fixedDims])
      let scaleRepeated = tf.broadcastTo(scale, [x.shape[0], ...fixedDims])

      while(shiftRepeated.shape.length < x.shape.length){
          shiftRepeated = tf.expandDims(shiftRepeated, -2);
          scaleRepeated = tf.expandDims(scaleRepeated, -2);
      }
      return tf.add(tf.mul(x, tf.add(1, scaleRepeated)), shiftRepeated)
    }
}


class SpatioTemporalDiTBlock extends tf.layers.Layer {
    constructor(hiddenSize, numHeads, mlpRatio = 4.0, isCausal = true, spatialRotaryEmb = null, temporalRotaryEmb = null) {
        super({});
        this.isCausal = isCausal;
        this.hiddenSize = hiddenSize;
        this.numHeads = numHeads;
        this.mlpRatio = mlpRatio;
        this.mlpHiddenDim = Math.floor(hiddenSize * mlpRatio);
        this.spatialRotaryEmb = spatialRotaryEmb;
        this.temporalRotaryEmb = temporalRotaryEmb;
        
        this.sNorm1 = tf.layers.layerNormalization({epsilon: 1e-6, trainable:false});
        this.sAttn = new SpatialAxialAttention(hiddenSize, numHeads, hiddenSize / numHeads, spatialRotaryEmb);
        this.sNorm2 = tf.layers.layerNormalization({epsilon: 1e-6, trainable:false});
         this.sMlp = [
             tf.layers.dense({ units: this.mlpHiddenDim, useBias: true, activation:'gelu', kernelInitializer: 'glorotUniform', biasInitializer: 'zeros'}),
             tf.layers.dense({ units: hiddenSize, useBias: true, kernelInitializer: 'glorotUniform', biasInitializer: 'zeros'})
         ]
        this.sAdaLNModulation = [
            tf.layers.activation({activation: 'silu'}),
            tf.layers.dense({units: 6 * hiddenSize, useBias: true, kernelInitializer: tf.initializers.zeros(), biasInitializer: tf.initializers.zeros()})
        ]

        this.tNorm1 = tf.layers.layerNormalization({epsilon: 1e-6, trainable:false});
        this.tAttn = new TemporalAxialAttention(hiddenSize, numHeads, hiddenSize / numHeads, isCausal, temporalRotaryEmb);
        this.tNorm2 = tf.layers.layerNormalization({epsilon: 1e-6, trainable:false});
          this.tMlp = [
            tf.layers.dense({ units: this.mlpHiddenDim, useBias: true, activation:'gelu', kernelInitializer: 'glorotUniform', biasInitializer: 'zeros'}),
            tf.layers.dense({ units: hiddenSize, useBias: true, kernelInitializer: 'glorotUniform', biasInitializer: 'zeros'})
          ]
        this.tAdaLNModulation = [
            tf.layers.activation({activation: 'silu'}),
            tf.layers.dense({units: 6 * hiddenSize, useBias: true, kernelInitializer: tf.initializers.zeros(), biasInitializer: tf.initializers.zeros()})
        ]
    }

    call(x, c) {
        let B = x.shape[0];
        let T = x.shape[1];
        let H = x.shape[2];
        let W = x.shape[3];
        let D = x.shape[4];


        // spatial block
        let sModulation = this.sAdaLNModulation[0].apply(c)
        sModulation = this.sAdaLNModulation[1].apply(sModulation)
        const [sShiftMsa, sScaleMsa, sGateMsa, sShiftMlp, sScaleMlp, sGateMlp] = tf.split(sModulation, 6, -1);

        let sAttnInput = this.modulate(this.sNorm1.apply(x), sShiftMsa, sScaleMsa);
        x = tf.add(x, this.gate(this.sAttn.apply(sAttnInput), sGateMsa));

        let sMlpInput = this.modulate(this.sNorm2.apply(x), sShiftMlp, sScaleMlp);
        let mlpOut = this.sMlp[0].apply(sMlpInput);
        mlpOut = this.sMlp[1].apply(mlpOut);

        x = tf.add(x, this.gate(mlpOut, sGateMlp));

        // temporal block
         let tModulation = this.tAdaLNModulation[0].apply(c)
         tModulation = this.tAdaLNModulation[1].apply(tModulation)
         const [tShiftMsa, tScaleMsa, tGateMsa, tShiftMlp, tScaleMlp, tGateMlp] = tf.split(tModulation, 6, -1);

         let tAttnInput = this.modulate(this.tNorm1.apply(x), tShiftMsa, tScaleMsa);
        x = tf.add(x, this.gate(this.tAttn.apply(tAttnInput), tGateMsa));

         let tMlpInput = this.modulate(this.tNorm2.apply(x), tShiftMlp, tScaleMlp);
        let tMlpOut = this.tMlp[0].apply(tMlpInput);
        tMlpOut = this.tMlp[1].apply(tMlpOut);
        x = tf.add(x, this.gate(tMlpOut, tGateMlp));
        return x;
    }

    modulate(x, shift, scale) {
      let fixedDims = Array(shift.shape.length - 1).fill(1)
      let shiftRepeated = tf.broadcastTo(shift, [x.shape[0], ...fixedDims])
      let scaleRepeated = tf.broadcastTo(scale, [x.shape[0], ...fixedDims])
      while(shiftRepeated.shape.length < x.shape.length){
          shiftRepeated = tf.expandDims(shiftRepeated, -2);
          scaleRepeated = tf.expandDims(scaleRepeated, -2);
      }
       return tf.add(tf.mul(x, tf.add(1, scaleRepeated)), shiftRepeated)
    }

    gate(x, g){
        let fixedDims = Array(g.shape.length - 1).fill(1)
        let gRepeated = tf.broadcastTo(g, [x.shape[0], ...fixedDims]);
        while(gRepeated.shape.length < x.shape.length){
             gRepeated = tf.expandDims(gRepeated, -2)
        }
       return tf.mul(gRepeated, x);
    }
}

class DiT extends tf.layers.Layer {
    constructor(
        inputH = 18,
        inputW = 32,
        patchSize = 2,
        inChannels = 16,
        hiddenSize = 1024,
        depth = 12,
        numHeads = 16,
        mlpRatio = 4.0,
        externalCondDim = 25,
        maxFrames = 32
    ) {
        super({});
        this.inputH = inputH;
        this.inputW = inputW;
        this.patchSize = patchSize;
        this.inChannels = inChannels;
        this.outChannels = inChannels;
        this.hiddenSize = hiddenSize;
        this.depth = depth;
        this.numHeads = numHeads;
        this.mlpRatio = mlpRatio;
        this.externalCondDim = externalCondDim;
        this.maxFrames = maxFrames;

        this.xEmbedder = new PatchEmbed(inputH, inputW, patchSize, inChannels, hiddenSize, false);
        this.tEmbedder = new TimestepEmbedder(hiddenSize);
        this.spatialRotaryEmb = new RotaryEmbedding(hiddenSize / numHeads / 2, null, 'pixel', null, 256);
        this.temporalRotaryEmb = new RotaryEmbedding(hiddenSize / numHeads);
        this.externalCond = externalCondDim > 0 ? tf.layers.dense({units: hiddenSize}) : tf.layers.identity();
        this.blocks = [];
          for (let i = 0; i < depth; i++) {
            this.blocks.push(
              new SpatioTemporalDiTBlock(
                  hiddenSize,
                  numHeads,
                  mlpRatio,
                  true,
                  this.spatialRotaryEmb,
                  this.temporalRotaryEmb,
              )
            );
          }
        this.finalLayer = new FinalLayer(hiddenSize, patchSize, this.outChannels);
    }


    call(x, t, externalCond) {
        let B = x.shape[0];
        let T = x.shape[1];
        let C = x.shape[2];
        let H = x.shape[3];
        let W = x.shape[4];

        // add spatial embeddings
        x = tf.reshape(x, [B * T, C, H, W]);
        x = this.xEmbedder.apply(x);
         x = tf.reshape(x, [B, T, this.xEmbedder.gridSize[0], this.xEmbedder.gridSize[1], this.hiddenSize]);

        t = tf.reshape(t, [B * T]);
         let c = this.tEmbedder.apply(t);
         c = tf.reshape(c, [B, T, this.hiddenSize]);
        if (externalCond) {
          c = tf.add(c, this.externalCond.apply(externalCond))
        }

        for (let block of this.blocks) {
          x = block.apply(x, c);
        }
        x = this.finalLayer.apply(x, c);
        x = tf.reshape(x, [B * T, this.xEmbedder.gridSize[0], this.xEmbedder.gridSize[1], this.patchSize * this.patchSize * this.outChannels]);
        x = this.unpatchify(x);
        x = tf.reshape(x, [B, T, this.outChannels, this.inputH, this.inputW])
        return x;
    }

    unpatchify(x) {
        const c = this.outChannels;
        const p = this.patchSize;
        const h = x.shape[1];
        const w = x.shape[2];


        x = tf.reshape(x, [x.shape[0], h, w, p, p, c]);
        x = tf.transpose(x, [0, 4, 1, 3, 2, 5]);
         let imgs = tf.reshape(x, [x.shape[0], c, h * p, w * p]);
        return imgs;
    }
}

export { DiT };