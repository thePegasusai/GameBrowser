/**
 * @fileoverview Browser-optimized Variational Autoencoder (VAE) implementation for video frame processing
 * @version 1.0.0
 * @license MIT
 */

import * as tf from '@tensorflow/tfjs-core'; // v4.x
import * as layers from '@tensorflow/tfjs-layers'; // v4.x
import { webgl } from '@tensorflow/tfjs-backend-webgl'; // v4.x
import { VAEConfig } from '../../types/model';
import { TensorSpec } from '../../types/tensor';
import { TensorOperations } from '../tensor/operations';
import { MEMORY_CONSTRAINTS, PERFORMANCE_THRESHOLDS } from '../../constants/model';

/**
 * Browser-optimized Variational Autoencoder for video frame encoding/decoding
 * Maintains <4GB memory usage and <50ms inference time through WebGL acceleration
 */
export class VAE {
    private encoder: tf.LayersModel;
    private decoder: tf.LayersModel;
    private readonly tensorOps: TensorOperations;
    private readonly performanceMonitor: PerformanceObserver;
    private readonly memoryUsage: { current: number; peak: number };
    private readonly webglContext: WebGLRenderingContext;

    constructor(
        private readonly config: VAEConfig,
        tensorOps: TensorOperations
    ) {
        this.tensorOps = tensorOps;
        this.memoryUsage = { current: 0, peak: 0 };

        // Initialize WebGL context
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2');
        if (!gl) {
            throw new Error('WebGL 2.0 is required for VAE operations');
        }
        this.webglContext = gl;

        // Configure TensorFlow.js for optimal performance
        tf.env().set('WEBGL_VERSION', 2);
        tf.env().set('WEBGL_FORCE_F16_TEXTURES', false);
        tf.env().set('WEBGL_PACK', true);

        // Initialize performance monitoring
        this.performanceMonitor = new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
                if (entry.duration > PERFORMANCE_THRESHOLDS.MAX_INFERENCE_TIME) {
                    console.warn(`VAE operation exceeded time threshold: ${entry.duration}ms`);
                }
            }
        });
        this.performanceMonitor.observe({ entryTypes: ['measure'] });

        // Build encoder and decoder models
        this.buildModels();
    }

    /**
     * Encodes input frames to latent space representation
     * @param input Input tensor of shape [batch, height, width, channels]
     * @returns Promise resolving to latent space tensor
     */
    public async encode(input: tf.Tensor4D): Promise<tf.Tensor> {
        const startTime = performance.now();
        performance.mark('encode-start');

        try {
            // Validate input tensor
            if (input.shape.length !== 4) {
                throw new Error('Input tensor must be 4D [batch, height, width, channels]');
            }

            // Track memory usage
            this.updateMemoryUsage();
            if (this.memoryUsage.current > MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE) {
                await this.cleanupMemory();
            }

            // Process input through encoder with WebGL acceleration
            const encoded = tf.tidy(() => {
                // Normalize input to [-1, 1]
                const normalized = tf.div(tf.sub(input, 127.5), 127.5);
                
                // Encode to latent space
                const encoded = this.encoder.predict(normalized) as tf.Tensor[];
                const [mean, logVar] = encoded;

                // Sample from latent distribution
                const std = tf.exp(tf.mul(logVar, 0.5));
                const eps = tf.randomNormal(mean.shape);
                return tf.add(mean, tf.mul(std, eps));
            });

            performance.mark('encode-end');
            performance.measure('VAE-encode', 'encode-start', 'encode-end');

            const duration = performance.now() - startTime;
            if (duration > PERFORMANCE_THRESHOLDS.MAX_INFERENCE_TIME) {
                console.warn(`Encoding time exceeded threshold: ${duration}ms`);
            }

            return encoded;

        } catch (error) {
            console.error('Error in VAE encoding:', error);
            throw error;
        }
    }

    /**
     * Decodes latent space representation to frames
     * @param latents Latent space tensor
     * @returns Promise resolving to reconstructed frame tensor
     */
    public async decode(latents: tf.Tensor): Promise<tf.Tensor> {
        performance.mark('decode-start');

        try {
            // Validate latent tensor
            if (latents.shape[latents.shape.length - 1] !== this.config.latentDim) {
                throw new Error(`Invalid latent dimension: ${latents.shape}`);
            }

            // Track memory usage
            this.updateMemoryUsage();
            if (this.memoryUsage.current > MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE) {
                await this.cleanupMemory();
            }

            // Process through decoder with WebGL acceleration
            const decoded = tf.tidy(() => {
                const output = this.decoder.predict(latents) as tf.Tensor;
                // Denormalize output from [-1, 1] to [0, 255]
                return tf.add(tf.mul(output, 127.5), 127.5);
            });

            performance.mark('decode-end');
            performance.measure('VAE-decode', 'decode-start', 'decode-end');

            return decoded;

        } catch (error) {
            console.error('Error in VAE decoding:', error);
            throw error;
        }
    }

    /**
     * Builds encoder and decoder models with WebGL optimization
     * @private
     */
    private buildModels(): void {
        // Build encoder
        const encoderInput = layers.input({ shape: this.config.inputSpec.shape.slice(1) });
        let x = encoderInput;

        // Encoder convolutional layers
        for (const filters of this.config.encoderLayers) {
            x = layers.conv2d({
                filters,
                kernelSize: 3,
                strides: 2,
                padding: 'same',
                activation: 'relu',
                kernelInitializer: 'glorotNormal'
            }).apply(x) as tf.Tensor;
        }

        // Latent space projections
        const flattenedShape = x.shape.slice(1).reduce((a, b) => a * b);
        const flattened = layers.flatten().apply(x);
        const mean = layers.dense({ units: this.config.latentDim }).apply(flattened);
        const logVar = layers.dense({ units: this.config.latentDim }).apply(flattened);

        this.encoder = tf.model({
            inputs: encoderInput,
            outputs: [mean, logVar],
            name: 'vae_encoder'
        });

        // Build decoder
        const decoderInput = layers.input({ shape: [this.config.latentDim] });
        let y = layers.dense({
            units: flattenedShape,
            activation: 'relu'
        }).apply(decoderInput);

        // Reshape to match encoder output
        y = layers.reshape({ targetShape: x.shape.slice(1) }).apply(y);

        // Decoder transposed convolution layers
        for (const filters of this.config.decoderLayers.slice().reverse()) {
            y = layers.conv2dTranspose({
                filters,
                kernelSize: 3,
                strides: 2,
                padding: 'same',
                activation: 'relu',
                kernelInitializer: 'glorotNormal'
            }).apply(y) as tf.Tensor;
        }

        // Output layer
        const decoderOutput = layers.conv2d({
            filters: 3,
            kernelSize: 3,
            padding: 'same',
            activation: 'tanh'
        }).apply(y);

        this.decoder = tf.model({
            inputs: decoderInput,
            outputs: decoderOutput,
            name: 'vae_decoder'
        });
    }

    /**
     * Updates current memory usage tracking
     * @private
     */
    private updateMemoryUsage(): void {
        const info = tf.memory();
        this.memoryUsage.current = info.numBytes;
        this.memoryUsage.peak = Math.max(this.memoryUsage.peak, info.numBytes);
    }

    /**
     * Cleans up unused tensors and WebGL resources
     * @private
     */
    private async cleanupMemory(): Promise<void> {
        await tf.dispose([]);
        this.webglContext.getExtension('WEBGL_lose_context')?.loseContext();
        this.webglContext.getExtension('WEBGL_lose_context')?.restoreContext();
        this.memoryUsage.current = 0;
    }

    /**
     * Returns current performance metrics
     */
    public getPerformanceMetrics(): {
        memoryUsage: { current: number; peak: number };
        webglMemory: number;
    } {
        return {
            memoryUsage: { ...this.memoryUsage },
            webglMemory: webgl.numBytesInGPU
        };
    }
}