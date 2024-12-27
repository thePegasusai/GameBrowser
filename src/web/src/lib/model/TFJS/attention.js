// attention.js

import * as tf from '@tensorflow/tfjs';
import { Timesteps, Positions2d } from './embeddings.js';
import {RotaryEmbedding} from './rotary_embedding.js'

class TemporalAxialAttention extends tf.layers.Layer {
    constructor(dim, heads = 4, dimHead = 32, isCausal = true, rotaryEmb = null) {
        super({});
        this.innerDim = dimHead * heads;
        this.heads = heads;
        this.dimHead = dimHead;
        this.isCausal = isCausal;
        this.toQKV = tf.layers.dense({ units: this.innerDim * 3, useBias: false });
        this.toOut = tf.layers.dense({ units: dim });

        this.rotaryEmb = rotaryEmb;
        this.timePosEmbedding = rotaryEmb == null ? new TimePosEmbedding(dim) : null;
    }


    call(x) {
        let B = x.shape[0];
        let T = x.shape[1];
        let H = x.shape[2];
        let W = x.shape[3];
        let D = x.shape[4];


        if (this.timePosEmbedding) {
          const timeEmb = this.timePosEmbedding.forward(tf.range(T).toFloat());
          x = tf.add(x, tf.reshape(timeEmb, [1, T, 1, 1, D]));
        }


        let qkv = this.toQKV.apply(x);
        qkv = tf.reshape(qkv, [B, T, H, W, 3, this.heads, this.dimHead]);
        qkv = tf.transpose(qkv, [4, 0, 2, 3, 5, 1, 6]);
        const q = qkv.slice([0, 0, 0, 0, 0, 0, 0], [1, B, H, W, this.heads, T, this.dimHead]).squeeze([0]);
        const k = qkv.slice([1, 0, 0, 0, 0, 0, 0], [1, B, H, W, this.heads, T, this.dimHead]).squeeze([0]);
        const v = qkv.slice([2, 0, 0, 0, 0, 0, 0], [1, B, H, W, this.heads, T, this.dimHead]).squeeze([0]);

        let reshapedQ = tf.reshape(q, [B * H * W, this.heads, T, this.dimHead]);
        let reshapedK = tf.reshape(k, [B * H * W, this.heads, T, this.dimHead]);
        let reshapedV = tf.reshape(v, [B * H * W, this.heads, T, this.dimHead]);

        if(this.rotaryEmb){
             reshapedQ = this.rotaryEmb.rotateQueriesOrKeys(reshapedQ, this.rotaryEmb.freqs, -2);
             reshapedK = this.rotaryEmb.rotateQueriesOrKeys(reshapedK, this.rotaryEmb.freqs, -2);
        }

        let attn = this.scaledDotProductAttention(reshapedQ, reshapedK, reshapedV, this.isCausal);
        attn = tf.reshape(attn, [B, H, W, this.heads, T, this.dimHead]);
        attn = tf.transpose(attn, [0, 4, 1, 2, 3, 5]);
        attn = tf.reshape(attn, [B, T, H, W, this.innerDim]);


        let out = this.toOut.apply(attn);
        return out;
    }

    scaledDotProductAttention(q, k, v, isCausal) {
        const qShape = q.shape;
        const kShape = k.shape;

        const dK = kShape[kShape.length - 1];

        let scores = tf.matMul(q, tf.transpose(k, [0, 1, 3, 2]));
        scores = tf.div(scores, Math.sqrt(dK));

        if (isCausal) {
          const mask = tf.linalg.bandPart(tf.ones([q.shape[2], k.shape[2]]), 0, -1);
          scores = tf.add(scores, tf.mul(tf.sub(mask, 1), -1e9))
        }
        const attn = tf.softmax(scores, -1);
        const out = tf.matMul(attn, v);
        return out;
    }
}


class SpatialAxialAttention extends tf.layers.Layer {
    constructor(dim, heads = 4, dimHead = 32, rotaryEmb = null) {
        super({});
        this.innerDim = dimHead * heads;
        this.heads = heads;
        this.dimHead = dimHead;

        this.toQKV = tf.layers.dense({ units: this.innerDim * 3, useBias: false });
        this.toOut = tf.layers.dense({ units: dim });
        this.rotaryEmb = rotaryEmb
        this.spacePosEmbedding = rotaryEmb == null ?  new SpacePosEmbedding(dim) : null
    }

    call(x) {
        let B = x.shape[0];
        let T = x.shape[1];
        let H = x.shape[2];
        let W = x.shape[3];
        let D = x.shape[4];

        if(this.spacePosEmbedding){
          const hSteps = tf.range(H).toFloat();
          const wSteps = tf.range(W).toFloat();
          const [hGrid, wGrid] = tf.meshgrid(hSteps, wSteps);
          const spaceEmb = this.spacePosEmbedding.forward([hGrid, wGrid]);
          x = tf.add(x, tf.reshape(spaceEmb, [1, 1, H, W, D]))
        }

        let qkv = this.toQKV.apply(x);
        qkv = tf.reshape(qkv, [B, T, H, W, 3, this.heads, this.dimHead]);
        qkv = tf.transpose(qkv, [4, 0, 1, 5, 2, 3, 6]);

        const q = qkv.slice([0, 0, 0, 0, 0, 0, 0], [1, B, T, this.heads, H, W, this.dimHead]).squeeze([0]);
        const k = qkv.slice([1, 0, 0, 0, 0, 0, 0], [1, B, T, this.heads, H, W, this.dimHead]).squeeze([0]);
        const v = qkv.slice([2, 0, 0, 0, 0, 0, 0], [1, B, T, this.heads, H, W, this.dimHead]).squeeze([0]);


        let reshapedQ = tf.reshape(q, [B * T, this.heads, H, W, this.dimHead]);
        let reshapedK = tf.reshape(k, [B * T, this.heads, H, W, this.dimHead]);
        let reshapedV = tf.reshape(v, [B * T, this.heads, H, W, this.dimHead]);

        if(this.rotaryEmb){
           let freqs = this.rotaryEmb.getAxialFreqs(H, W)
           let [rotatedQ, rotatedK] = this.rotaryEmb.rotateQueriesAndKeys(reshapedQ, reshapedK, freqs);
           reshapedQ = rotatedQ;
           reshapedK = rotatedK;
        }
        
        let reshapedQForAttn = tf.reshape(reshapedQ, [B*T, this.heads, H * W, this.dimHead]);
        let reshapedKForAttn = tf.reshape(reshapedK, [B*T, this.heads, H * W, this.dimHead]);
        let reshapedVForAttn = tf.reshape(reshapedV, [B*T, this.heads, H * W, this.dimHead]);

        let attn = this.scaledDotProductAttention(reshapedQForAttn, reshapedKForAttn, reshapedVForAttn);


        attn = tf.reshape(attn, [B, T, this.heads, H, W, this.dimHead]);
        attn = tf.transpose(attn, [0, 1, 3, 4, 2, 5]);
        attn = tf.reshape(attn, [B, T, H, W, this.innerDim]);

        let out = this.toOut.apply(attn);
        return out;
    }

     scaledDotProductAttention(q, k, v, isCausal) {
         const qShape = q.shape;
        const kShape = k.shape;

        const dK = kShape[kShape.length - 1];

        let scores = tf.matMul(q, tf.transpose(k, [0, 1, 3, 2]));
        scores = tf.div(scores, Math.sqrt(dK));

        if (isCausal) {
          const mask = tf.linalg.bandPart(tf.ones([q.shape[2], k.shape[2]]), 0, -1);
          scores = tf.add(scores, tf.mul(tf.sub(mask, 1), -1e9))
        }

        const attn = tf.softmax(scores, -1);
        const out = tf.matMul(attn, v);
        return out;
    }
}

class TimePosEmbedding {
     constructor(dim) {
        this.timePosEmbedding = new Timesteps(dim);
    }
    forward(t){
        return this.timePosEmbedding.forward(t)
    }
}

class SpacePosEmbedding {
    constructor(dim) {
        this.spacePosEmbedding = new Positions2d(dim);
    }
    forward(grid){
      return this.spacePosEmbedding.forward(grid);
    }
}



export { TemporalAxialAttention, SpatialAxialAttention };