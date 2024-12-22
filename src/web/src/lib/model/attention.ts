/**
 * @fileoverview Memory-efficient multi-head self-attention implementation for browser-based video game diffusion model
 * @version 1.0.0
 * @license MIT
 */

import * as tf from '@tensorflow/tfjs-core'; // v4.x
import { TensorOperations } from '../tensor/operations';
import { ModelConfig } from '../../types/model';
import { validateTensorOperations } from '../utils/validation';
import { PERFORMANCE_THRESHOLDS } from '../../constants/model';

/**
 * Implements memory-efficient multi-head self-attention mechanism for the DiT model
 * Optimized for browser execution with WebGL acceleration
 */
export class MultiHeadAttention {
    private readonly numHeads: number;
    private readonly headDim: number;
    private readonly hiddenSize: number;
    private readonly attentionDropout: number;
    private readonly tensorOps: TensorOperations;
    private readonly scaleFactor: number;

    // Attention weights
    private queryWeight: tf.Tensor;
    private keyWeight: tf.Tensor;
    private valueWeight: tf.Tensor;
    private outputWeight: tf.Tensor;

    constructor(config: ModelConfig, tensorOps: TensorOperations) {
        this.numHeads = config.ditConfig?.numHeads || 8;
        this.hiddenSize = config.ditConfig?.hiddenSize || 512;
        this.headDim = this.hiddenSize / this.numHeads;
        this.attentionDropout = config.ditConfig?.dropoutRate || 0.1;
        this.tensorOps = tensorOps;
        this.scaleFactor = Math.sqrt(this.headDim);

        // Initialize attention weights with proper shapes
        this.initializeWeights();
    }

    /**
     * Computes multi-head self-attention with optimized memory usage
     * @param input Input tensor of shape [batch, seq_len, hidden_size]
     * @param training Whether the model is in training mode
     * @returns Attention output tensor of shape [batch, seq_len, hidden_size]
     */
    public async computeAttention(
        input: tf.Tensor,
        training: boolean = false
    ): Promise<tf.Tensor> {
        const startTime = performance.now();

        try {
            // Validate input tensor
            const validation = validateTensorOperations(
                { shape: input.shape, dtype: input.dtype },
                'attention'
            );
            if (!validation.isValid) {
                throw new Error(`Invalid input tensor: ${validation.errors.join(', ')}`);
            }

            // Compute query, key, value projections with memory optimization
            const [query, key, value] = await Promise.all([
                this.tensorOps.batchProcess(
                    tf.matMul(input, this.queryWeight),
                    { shape: [input.shape[0], input.shape[1], this.hiddenSize] },
                    { useWebGL: true }
                ),
                this.tensorOps.batchProcess(
                    tf.matMul(input, this.keyWeight),
                    { shape: [input.shape[0], input.shape[1], this.hiddenSize] },
                    { useWebGL: true }
                ),
                this.tensorOps.batchProcess(
                    tf.matMul(input, this.valueWeight),
                    { shape: [input.shape[0], input.shape[1], this.hiddenSize] },
                    { useWebGL: true }
                )
            ]);

            // Split heads with memory optimization
            const splitQuery = await this.splitHeads(query);
            const splitKey = await this.splitHeads(key);
            const splitValue = await this.splitHeads(value);

            // Compute scaled dot-product attention
            const attentionScores = tf.tidy(() => {
                const scores = tf.matMul(splitQuery, splitKey, false, true);
                return tf.div(scores, tf.scalar(this.scaleFactor));
            });

            // Apply attention dropout during training
            const attentionWeights = training ? 
                tf.dropout(tf.softmax(attentionScores), this.attentionDropout) :
                tf.softmax(attentionScores);

            // Compute weighted sum
            const attentionOutput = tf.tidy(() => {
                const weighted = tf.matMul(attentionWeights, splitValue);
                return this.combineHeads(weighted);
            });

            // Project to output dimension
            const output = await this.tensorOps.batchProcess(
                tf.matMul(attentionOutput, this.outputWeight),
                { shape: [input.shape[0], input.shape[1], this.hiddenSize] },
                { useWebGL: true }
            );

            // Check performance threshold
            const processingTime = performance.now() - startTime;
            if (processingTime > PERFORMANCE_THRESHOLDS.MAX_INFERENCE_TIME) {
                console.warn(`Attention computation exceeded time threshold: ${processingTime}ms`);
            }

            // Clean up intermediate tensors
            tf.dispose([
                query, key, value,
                splitQuery, splitKey, splitValue,
                attentionScores, attentionWeights, attentionOutput
            ]);

            return output;

        } catch (error) {
            console.error('Error in attention computation:', error);
            throw error;
        }
    }

    /**
     * Splits input tensor into multiple attention heads
     * @param input Input tensor to split
     * @returns Reshaped tensor with multiple heads
     */
    private async splitHeads(input: tf.Tensor): Promise<tf.Tensor> {
        const batchSize = input.shape[0];
        const seqLength = input.shape[1];

        return this.tensorOps.reshape(
            input,
            [batchSize, seqLength, this.numHeads, this.headDim],
            { useWebGL: true }
        );
    }

    /**
     * Combines multiple attention heads back into original dimension
     * @param input Input tensor with multiple heads
     * @returns Combined tensor
     */
    private async combineHeads(input: tf.Tensor): Promise<tf.Tensor> {
        const batchSize = input.shape[0];
        const seqLength = input.shape[1];

        return this.tensorOps.reshape(
            input,
            [batchSize, seqLength, this.hiddenSize],
            { useWebGL: true }
        );
    }

    /**
     * Initializes attention weights with proper shapes
     * @private
     */
    private initializeWeights(): void {
        // Initialize query, key, value projection matrices
        this.queryWeight = tf.variable(
            tf.randomNormal([this.hiddenSize, this.hiddenSize], 0, 0.02)
        );
        this.keyWeight = tf.variable(
            tf.randomNormal([this.hiddenSize, this.hiddenSize], 0, 0.02)
        );
        this.valueWeight = tf.variable(
            tf.randomNormal([this.hiddenSize, this.hiddenSize], 0, 0.02)
        );
        this.outputWeight = tf.variable(
            tf.randomNormal([this.hiddenSize, this.hiddenSize], 0, 0.02)
        );
    }

    /**
     * Cleans up attention weights and resources
     */
    public dispose(): void {
        tf.dispose([
            this.queryWeight,
            this.keyWeight,
            this.valueWeight,
            this.outputWeight
        ]);
    }
}