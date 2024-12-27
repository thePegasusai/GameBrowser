// rotary_embedding.js

import * as tf from '@tensorflow/tfjs';

class RotaryEmbedding {
  constructor(dim, customFreqs = null, freqsFor = 'lang', theta = 10000, maxFreq = 10, numFreqs = 1, learnedFreq = false, useXpos = false, xposScaleBase = 512, interpolateFactor = 1., thetaRescaleFactor = 1., seqBeforeHeadDim = false, cacheIfPossible = true, cacheMaxSeqLen = 8192) {
    this.dim = dim;
    this.theta = theta * Math.pow(thetaRescaleFactor, dim / (dim - 2));
    this.freqsFor = freqsFor;
    this.maxFreq = maxFreq;
    this.learnedFreq = learnedFreq;
    this.useXpos = useXpos;
    this.xposScaleBase = xposScaleBase;
    this.interpolateFactor = interpolateFactor;
    this.seqBeforeHeadDim = seqBeforeHeadDim;
    this.cacheIfPossible = cacheIfPossible;
    this.cacheMaxSeqLen = cacheMaxSeqLen;
    this.numFreqs = numFreqs
    
    
    if (customFreqs) {
      this.freqs = tf.tensor(customFreqs);
    } else if (freqsFor == 'lang') {
        const seq = tf.range(0, dim, 2).toFloat();
        this.freqs = tf.pow(this.theta, seq.div(dim).mul(-1));
    } else if (freqsFor == 'pixel') {
        this.freqs = tf.linspace(1, maxFreq / 2, dim / 2).mul(Math.PI)
    } else if (freqsFor == 'constant') {
      this.freqs = tf.ones([numFreqs]);
    }
    this.freqs = this.learnedFreq ? tf.variable(this.freqs) : this.freqs;


    if(freqsFor == 'spacetime'){
      const seq = tf.range(0, dim, 2).toFloat();
      this.timeFreqs = tf.pow(this.theta, seq.div(dim).mul(-1));
      this.timeFreqs = this.learnedFreq ? tf.variable(this.timeFreqs) : this.timeFreqs;
    }
    this.cachedFreqs = tf.zeros([cacheMaxSeqLen, dim]);
    this.cachedFreqsSeqLen = 0;


    if(useXpos){
        this.scale = (tf.range(0, dim, 2).add(0.4 * dim)).div(1.4 * dim);
        this.cachedScales = tf.zeros([cacheMaxSeqLen, dim]);
        this.cachedScalesSeqLen = 0
    }
  }

  getSeqPos(seqLen, dtype, offset = 0){
    return tf.range(seqLen, dtype).add(offset).div(this.interpolateFactor)
  }

  rotateQueriesOrKeys(t, freqs, seqDim, offset = 0, scale = null) {
    if(!seqDim) seqDim = this.seqBeforeHeadDim ? -3 : -2;
    
    if(!this.useXpos || scale != null){
      let seqLen = t.shape[seqDim];
      let seq = this.getSeqPos(seqLen, t.dtype, offset)
      let seqFreqs = this.forward(seq, freqs, seqLen, offset);
      if(seqDim == -3)
        seqFreqs = tf.reshape(seqFreqs, [seqLen, 1, this.dim]);
      return this.applyRotaryEmb(seqFreqs, t, scale, seqDim);
    } else {
      throw new Error('for length extrapolatable rotary embeddings, you need to use `.rotateQueriesAndKeys` method instead and pass in both queries and keys')
    }
  }

  applyRotaryEmb(freqs, t, scale, seqDim){
    if (!scale) scale = 1.0;
    let rotDim = freqs.shape.slice(-1)[0];
    if(t.shape.length == 3 && seqDim == -2){
       seqLen = t.shape[seqDim];
       freqs = freqs.slice([-seqLen, 0], [seqLen, rotDim]);
    }
    let tLeft = t.slice([0], [t.shape.length -1], t.shape.slice(-1).slice(0, [rotDim]), [0]);
    let tMiddle = t.slice([0], [t.shape.length -1], [rotDim]);
    let tRight = t.slice([0], [t.shape.length -1], [0], [t.shape.slice(-1)[0] - rotDim])

    let cos = tf.cos(freqs).mul(scale)
    let sin = tf.sin(freqs).mul(scale)

    let tTransformed = tf.add(tf.mul(tMiddle, cos), tf.mul(this.rotateHalf(tMiddle), sin));

    let out = tLeft;
    if(tTransformed) out = tf.concat([out, tTransformed], -1);
    if(tRight) out = tf.concat([out, tRight], -1);
    return out;
  }

  getScale(t, seqLen, offset = 0){
      if (this.useXpos == false) return 1.
      let shouldCache = (this.cacheIfPossible && seqLen && offset + seqLen <= this.cacheMaxSeqLen)

      if (shouldCache && this.cachedScales && offset + seqLen <= this.cachedScalesSeqLen)
          return this.cachedScales.slice([offset, 0], [seqLen, this.dim])

      let power = (t.sub(t.shape[0] / 2)).div(this.xposScaleBase);
      let scale = tf.pow(this.scale, tf.reshape(power, [-1, 1]));
      scale = tf.reshape(scale, [-1, this.dim])

      if(shouldCache && offset == 0){
        this.cachedScales = this.cachedScales.slice();
        this.cachedScales.assign(tf.pad(scale.slice(), [[0,this.cacheMaxSeqLen-seqLen], [0,0]]));
        this.cachedScalesSeqLen = seqLen;
      }
      return scale;
  }

  rotateQueriesWithCachedKeys(q, k, seqDim, offset = 0){
      let qLen = q.shape[seqDim];
      let kLen = k.shape[seqDim];
      if(qLen > kLen) throw new Error('q length can not be bigger than k length');

      let qScale = 1.;
      let kScale = 1.;
      
      if(this.useXpos){
        let seq = this.getSeqPos(kLen, q.dtype)
        qScale = this.getScale(seq.slice([-qLen]), kLen).cast(q.dtype);
        kScale = this.getScale(seq, kLen).cast(k.dtype).pow(-1)
      }

      let rotatedQ = this.rotateQueriesOrKeys(q, null, seqDim, kLen - qLen + offset, qScale)
      let rotatedK = this.rotateQueriesOrKeys(k, null, seqDim, offset, kScale)
      return [rotatedQ, rotatedK]
  }

  rotateQueriesAndKeys(q, k, freqs, seqDim){
    if(!this.useXpos) throw new Error('can only use rotateQueriesAndKeys when xpos is enabled')
      if(!seqDim) seqDim = this.seqBeforeHeadDim ? -3 : -2;

      let seqLen = q.shape[seqDim];
      let seq = this.getSeqPos(seqLen, q.dtype)
      let seqFreqs = this.forward(seq, freqs, seqLen);
      let scale = this.getScale(seq, seqLen).cast(q.dtype)

      if(seqDim == -3){
        seqFreqs = tf.reshape(seqFreqs, [seqLen, 1, this.dim]);
        scale = tf.reshape(scale, [seqLen, 1, this.dim]);
      }
      let rotatedQ = this.applyRotaryEmb(seqFreqs, q, scale, seqDim)
      let rotatedK = this.applyRotaryEmb(seqFreqs, k, scale.pow(-1), seqDim)
      return [rotatedQ, rotatedK]
  }

  getAxialFreqs(...dims){
      let Colon = null;
      let allFreqs = [];

      for(let [ind, dim] of dims.entries()){
          let usePixel = (this.freqsFor == 'pixel' || this.freqsFor == 'spacetime') && ind >= dims.length - 2;
          let pos;
          if(usePixel){
            pos = tf.linspace(-1, 1, dim)
          } else {
            pos = tf.range(dim).toFloat();
          }
        let seqFreqs;
        if(this.freqsFor == 'spacetime' && !usePixel){
            seqFreqs = this.forward(pos, this.timeFreqs, dim);
          } else {
            seqFreqs = this.forward(pos, this.freqs, dim);
          }

        let allAxis = Array(dims.length).fill(Colon);
          allAxis[ind] = Colon;
          
        let newAxisSlice = [null, ...allAxis, null];
        let reshapedSeqFreqs = tf.reshape(seqFreqs, newAxisSlice.map(x => x == null ? 1 : x == Colon ? null : -1).filter(x => x != null) )
        allFreqs.push(reshapedSeqFreqs)
      }
     let freq = allFreqs[0];
     for(let i = 1; i < allFreqs.length; i++){
       freq = tf.broadcastTo(freq, allFreqs[i].shape)
       freq = tf.concat([freq, allFreqs[i]], -1)
     }
     return freq;
  }

  forward(t, freqs, seqLen, offset = 0) {
    let shouldCache = (this.cacheIfPossible && this.learnedFreq == false && seqLen && this.freqsFor != 'pixel' && offset + seqLen <= this.cacheMaxSeqLen);
    if (shouldCache && this.cachedFreqs && offset + seqLen <= this.cachedFreqsSeqLen) {
      return this.cachedFreqs.slice([offset,0],[seqLen, this.dim])
    }
      let newFreq = tf.mul(t.cast(freqs.dtype), freqs);
      newFreq = tf.reshape(newFreq, [-1, this.dim])
      if (shouldCache && offset == 0) {
        this.cachedFreqs = tf.pad(newFreq, [[0,this.cacheMaxSeqLen-seqLen], [0,0]])
        this.cachedFreqsSeqLen = seqLen;
      }
      return newFreq;
  }


    rotateHalf(x){
    let r = 2
    let xShape = x.shape;
    let d = xShape.slice(-1)[0] / r;
    let xReshaped = tf.reshape(x, [...xShape.slice(0, -1), d, r])

    let x1 = xReshaped.slice([0], [xReshaped.shape.length -1], [0], [1])
    let x2 = xReshaped.slice([0], [xReshaped.shape.length -1], [1], [1])
    let xStacked = tf.stack([-x2, x1], -1)
    return tf.reshape(xStacked, [...xShape.slice(0, -1), d*r])
  }
}

export { RotaryEmbedding };