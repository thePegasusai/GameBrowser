// utils.js

import * as tf from '@tensorflow/tfjs';

function sigmoidBetaSchedule(timesteps, start = -3, end = 3, tau = 1, clampMin = 1e-5) {
    const steps = timesteps + 1;
    const t = tf.linspace(0, timesteps, steps).toFloat().div(timesteps);
    const vStart = tf.sigmoid(tf.scalar(start / tau));
    const vEnd = tf.sigmoid(tf.scalar(end / tau));
    const alphasCumprod = tf.div(tf.sub(tf.sigmoid(tf.add(tf.mul(t, (end - start)), start).div(tau)).mul(-1), vEnd).add(vEnd) , tf.sub(vEnd, vStart));
    const normalizedAlphasCumprod =  tf.div(alphasCumprod, alphasCumprod.slice([0], [1]));

    const betas = tf.sub(1, tf.div(normalizedAlphasCumprod.slice([1]), normalizedAlphasCumprod.slice([0,-1])));
    return tf.clipByValue(betas, 0, 0.999);
}

class ActionEmbedder extends tf.layers.Layer{
    constructor(numCategories, embeddingDim) {
        super({})
        this.embeds = []
       for(let categorySize of numCategories) {
          this.embeds.push(tf.layers.embedding({inputDim: categorySize, outputDim: embeddingDim,  embeddingsInitializer: 'glorotUniform' }))
        }
        this.linear = tf.layers.dense({units: embeddingDim * numCategories.length, useBias: true, kernelInitializer: 'glorotUniform', biasInitializer: 'zeros'})
    }
    call(actions) {
        let B = actions.shape[0];
        let T = actions.shape[1];
        let numActions = actions.shape[2];
        let embeds = [];
         for(let i = 0; i < numActions; i++){
             embeds.push(this.embeds[i].apply(tf.cast(actions.slice([0], [-1], [0], [1]), 'int32')))
         }
        let combinedEmbeds = tf.concat(embeds, -1);
       return this.linear.apply(combinedEmbeds);
    }
}
 function createActionEmbedding(actions) {
     // Assumes actions is a list of dicts, similar to what was provided
    const numCategories = [4, 1]
    const embeddingDim = 25
    const embedder = new ActionEmbedder(numCategories, embeddingDim)
    const batchSize = actions.length;
    const seqLen = actions[0].length;
    const numActions = numCategories.length;
    const tensor = tf.zeros([batchSize, seqLen, numActions]);
     for (let i = 0; i < batchSize; i++){
         for (let t = 0; t < seqLen; t++){
            let action_dict = actions[i][t];
              if (action_dict["forward"] == 1.0) {
                 tensor = tensor.slice([0],[i],[t], [0], [1]).concat(tf.ones([1,1,1,1]).mul(0),  [0,0,0,1])
             } else if (action_dict["back"] == 1.0) {
                 tensor = tensor.slice([0],[i],[t], [0], [1]).concat(tf.ones([1,1,1,1]).mul(1), [0,0,0,1])
             } else if (action_dict["left"] == 1.0) {
                tensor = tensor.slice([0],[i],[t], [0], [1]).concat(tf.ones([1,1,1,1]).mul(2), [0,0,0,1])
             } else if (action_dict["right"] == 1.0) {
                 tensor = tensor.slice([0],[i],[t], [0], [1]).concat(tf.ones([1,1,1,1]).mul(3), [0,0,0,1])
            }
              //camera
            tensor = tensor.slice([0],[i],[t],[1], [1]).concat(tf.scalar(action_dict["camera"][0]).reshape([1,1,1,1]), [0,0,0,1])
        }
     }
    return embedder.apply(tensor);
}

export { sigmoidBetaSchedule, createActionEmbedding };