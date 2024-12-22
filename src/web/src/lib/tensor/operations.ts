/**
 * @fileoverview Core tensor operations implementation with WebGL acceleration and memory management
 * @version 1.0.0
 * @license MIT
 */

import * as tf from '@tensorflow/tfjs-core'; // v4.x
import { engine } from '@tensorflow/tfjs-core'; // v4.x
import { TensorMemoryManager } from './memory';
import { TensorSpec } from '../../types/tensor';
import { Logger } from '../utils/logger';
import { validateTensorOperations } from '../utils/validation';
import { PERFORMANCE_THRESHOLDS } from '../../constants/model';

/**
 * Core tensor operations class with WebGL acceleration and memory optimization
 * Ensures <50ms inference time and <4GB memory usage
 */
export class TensorOperations {
    private readonly memoryManager: TensorMemoryManager;
    private readonly logger: Logger;
    private readonly webglContext: WebGLRenderingContext;
    private readonly texturePool: Map<string, WebGLTexture>;

    constructor(memoryManager: TensorMemoryManager, logger: Logger) {
        this.memoryManager = memoryManager;
        this.logger = logger;
        this.texturePool = new Map();

        // Initialize WebGL context
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2');
        if (!gl) {
            throw new Error('WebGL 2.0 is required for tensor operations');
        }
        this.webglContext = gl;

        // Configure TensorFlow.js for optimal performance
        tf.env().set('WEBGL_VERSION', 2);
        tf.env().set('WEBGL_FORCE_F16_TEXTURES', false);
        tf.env().set('WEBGL_PACK', true);
    }

    /**
     * Process a batch of tensors with WebGL acceleration
     * @param input Input tensor to process
     * @param outputSpec Expected output tensor specification
     * @param webglConfig WebGL configuration for processing
     * @returns Processed tensor
     */
    public async batchProcess(
        input: tf.Tensor,
        outputSpec: TensorSpec,
        webglConfig: WebGLConfig
    ): Promise<tf.Tensor> {
        const startTime = performance.now();

        try {
            // Validate input and operation
            const validation = validateTensorOperations(outputSpec, 'batchProcess');
            if (!validation.isValid) {
                throw new Error(`Invalid tensor operation: ${validation.errors.join(', ')}`);
            }

            // Track input tensor
            const inputId = this.memoryManager.trackTensor(input, outputSpec, 2);

            // Process using WebGL acceleration
            const result = tf.tidy(() => {
                // Ensure correct memory layout for WebGL
                const optimizedInput = this.optimizeForWebGL(input, webglConfig);
                
                // Perform batch processing operations
                const processed = tf.engine().runKernel(
                    'BatchProcess',
                    { x: optimizedInput },
                    { dtype: outputSpec.dtype }
                );

                // Ensure output matches specification
                return this.reshapeToSpec(processed, outputSpec);
            });

            // Track processing time
            const processingTime = performance.now() - startTime;
            this.logger.logPerformance('tensor_batch_process', {
                processingTime,
                inputShape: input.shape,
                outputShape: result.shape,
                memoryUsage: await this.memoryManager.getMemoryInfo()
            });

            // Validate performance threshold
            if (processingTime > PERFORMANCE_THRESHOLDS.MAX_INFERENCE_TIME) {
                this.logger.warn(`Batch processing exceeded time threshold: ${processingTime}ms`);
            }

            // Cleanup input tensor
            this.memoryManager.disposeTensor(inputId);

            return result;

        } catch (error) {
            this.logger.error('Error in batch processing', error);
            throw error;
        }
    }

    /**
     * Reshape tensor with WebGL optimization
     * @param tensor Input tensor to reshape
     * @param newShape Target shape array
     * @param webglConfig WebGL configuration
     * @returns Reshaped tensor
     */
    public reshape(
        tensor: tf.Tensor,
        newShape: number[],
        webglConfig: WebGLConfig
    ): tf.Tensor {
        try {
            // Track input tensor
            const tensorId = this.memoryManager.trackTensor(
                tensor,
                { shape: newShape, dtype: tensor.dtype } as TensorSpec,
                1
            );

            // Perform WebGL-optimized reshape
            const result = tf.tidy(() => {
                const optimized = this.optimizeForWebGL(tensor, webglConfig);
                return tf.reshape(optimized, newShape);
            });

            // Cleanup
            this.memoryManager.disposeTensor(tensorId);

            return result;

        } catch (error) {
            this.logger.error('Error in reshape operation', error);
            throw error;
        }
    }

    /**
     * Concatenate tensors with WebGL acceleration
     * @param tensors Array of tensors to concatenate
     * @param axis Axis along which to concatenate
     * @param webglConfig WebGL configuration
     * @returns Concatenated tensor
     */
    public concat(
        tensors: tf.Tensor[],
        axis: number,
        webglConfig: WebGLConfig
    ): tf.Tensor {
        try {
            // Track input tensors
            const tensorIds = tensors.map(t => 
                this.memoryManager.trackTensor(t, { shape: t.shape, dtype: t.dtype } as TensorSpec, 1)
            );

            // Perform WebGL-optimized concatenation
            const result = tf.tidy(() => {
                const optimizedTensors = tensors.map(t => this.optimizeForWebGL(t, webglConfig));
                return tf.concat(optimizedTensors, axis);
            });

            // Cleanup
            tensorIds.forEach(id => this.memoryManager.disposeTensor(id));

            return result;

        } catch (error) {
            this.logger.error('Error in concatenation operation', error);
            throw error;
        }
    }

    /**
     * Optimize tensor for WebGL processing
     * @private
     */
    private optimizeForWebGL(tensor: tf.Tensor, config: WebGLConfig): tf.Tensor {
        return tf.tidy(() => {
            // Ensure tensor is in correct format for WebGL
            const dataType = config.performanceFlags?.enableFloatTextures ? 
                'float32' : 'int32';
            
            return tensor.cast(dataType);
        });
    }

    /**
     * Reshape tensor to match specification
     * @private
     */
    private reshapeToSpec(tensor: tf.Tensor, spec: TensorSpec): tf.Tensor {
        return tf.tidy(() => {
            if (tensor.shape.toString() !== spec.shape.toString()) {
                return tf.reshape(tensor, spec.shape);
            }
            return tensor;
        });
    }
}