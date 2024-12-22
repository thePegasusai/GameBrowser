/**
 * @fileoverview Core metrics collection and monitoring utility for browser-based video game diffusion model
 * @version 1.0.0
 * @license MIT
 */

import * as tf from '@tensorflow/tfjs-core'; // v4.x
import { Logger } from './logger';
import { validateMemoryConstraints } from './validation';
import { MEMORY_CONSTRAINTS, PERFORMANCE_THRESHOLDS } from '../../constants/model';

/**
 * Types for metrics collection and monitoring
 */
interface PerformanceMetrics {
    inferenceTime: number;
    memoryUsage: MemoryMetrics;
    gpuUtilization: number;
    fps: number;
    timestamp: number;
}

interface MemoryMetrics {
    heapUsed: number;
    heapTotal: number;
    gpuMemoryUsage: number;
    tensorCount: number;
    bufferCount: number;
    isWarning: boolean;
}

interface MetricThresholds {
    maxInferenceTime: number;
    maxMemoryUsage: number;
    maxGPUMemory: number;
    targetFPS: number;
}

interface MetricsConfig {
    retentionPeriodMs: number;
    thresholds: MetricThresholds;
    enableAlerts: boolean;
}

/**
 * Core metrics collection and monitoring class
 */
export class MetricsCollector {
    private metrics: Map<string, PerformanceMetrics[]>;
    private readonly retentionPeriodMs: number;
    private readonly logger: Logger;
    private readonly thresholds: MetricThresholds;
    private cleanupInterval: NodeJS.Timer;

    constructor(config: MetricsConfig) {
        this.metrics = new Map();
        this.retentionPeriodMs = config.retentionPeriodMs;
        this.thresholds = config.thresholds;
        this.logger = new Logger({
            level: 'info',
            namespace: 'metrics',
            persistLogs: true
        });

        // Initialize cleanup interval
        this.cleanupInterval = setInterval(
            () => this.cleanupOldMetrics(),
            MEMORY_CONSTRAINTS.MEMORY_CHECK_INTERVAL
        );
    }

    /**
     * Records and validates a new performance metric
     */
    public async recordMetric(
        name: string,
        value: number,
        type: 'inference' | 'memory' | 'gpu' | 'fps',
        context: Record<string, unknown> = {}
    ): Promise<void> {
        try {
            const memoryInfo = await this.getMemoryMetrics();
            const metric: PerformanceMetrics = {
                inferenceTime: type === 'inference' ? value : 0,
                memoryUsage: memoryInfo,
                gpuUtilization: type === 'gpu' ? value : 0,
                fps: type === 'fps' ? value : 0,
                timestamp: Date.now()
            };

            // Store metric
            if (!this.metrics.has(name)) {
                this.metrics.set(name, []);
            }
            this.metrics.get(name)!.push(metric);

            // Validate against thresholds
            await this.validateMetric(metric, name, context);

            // Log metric
            this.logger.logPerformance(`Metric ${name}: ${value}`, {
                type,
                context,
                memoryInfo
            });
        } catch (error) {
            this.logger.logError('Error recording metric', error as Error);
        }
    }

    /**
     * Retrieves metrics for analysis
     */
    public getMetrics(
        metricName: string,
        timeRangeMs?: number
    ): PerformanceMetrics[] {
        const metrics = this.metrics.get(metricName) || [];
        if (timeRangeMs) {
            const cutoffTime = Date.now() - timeRangeMs;
            return metrics.filter(m => m.timestamp >= cutoffTime);
        }
        return metrics;
    }

    /**
     * Measures execution time of an async operation
     */
    public async measureExecutionTime<T>(
        operation: () => Promise<T>,
        operationName: string
    ): Promise<{ result: T; duration: number }> {
        const start = performance.now();
        try {
            const result = await operation();
            const duration = performance.now() - start;

            await this.recordMetric(
                operationName,
                duration,
                'inference',
                { success: true }
            );

            return { result, duration };
        } catch (error) {
            const duration = performance.now() - start;
            await this.recordMetric(
                operationName,
                duration,
                'inference',
                { success: false, error }
            );
            throw error;
        }
    }

    /**
     * Tracks detailed memory usage
     */
    private async getMemoryMetrics(): Promise<MemoryMetrics> {
        const tfMemory = tf.memory();
        const memoryInfo = performance?.memory || {
            usedJSHeapSize: 0,
            totalJSHeapSize: 0
        };

        const metrics: MemoryMetrics = {
            heapUsed: memoryInfo.usedJSHeapSize / (1024 * 1024),
            heapTotal: memoryInfo.totalJSHeapSize / (1024 * 1024),
            gpuMemoryUsage: tfMemory.numBytesInGPU / (1024 * 1024),
            tensorCount: tfMemory.numTensors,
            bufferCount: tfMemory.numDataBuffers,
            isWarning: false
        };

        // Check memory warning threshold
        metrics.isWarning = metrics.heapUsed / metrics.heapTotal > 
            MEMORY_CONSTRAINTS.CLEANUP_THRESHOLD;

        return metrics;
    }

    /**
     * Validates metrics against thresholds
     */
    private async validateMetric(
        metric: PerformanceMetrics,
        name: string,
        context: Record<string, unknown>
    ): Promise<void> {
        // Validate inference time
        if (metric.inferenceTime > this.thresholds.maxInferenceTime) {
            this.logger.logAlert('High inference time detected', {
                metric: name,
                value: metric.inferenceTime,
                threshold: this.thresholds.maxInferenceTime,
                context
            });
        }

        // Validate memory usage
        if (metric.memoryUsage.isWarning) {
            this.logger.logAlert('High memory usage detected', {
                metric: name,
                usage: metric.memoryUsage,
                context
            });
        }

        // Validate GPU memory
        if (metric.memoryUsage.gpuMemoryUsage > this.thresholds.maxGPUMemory) {
            this.logger.logAlert('High GPU memory usage detected', {
                metric: name,
                usage: metric.memoryUsage.gpuMemoryUsage,
                threshold: this.thresholds.maxGPUMemory,
                context
            });
        }

        // Validate FPS
        if (metric.fps > 0 && metric.fps < this.thresholds.targetFPS) {
            this.logger.logAlert('Low FPS detected', {
                metric: name,
                value: metric.fps,
                target: this.thresholds.targetFPS,
                context
            });
        }
    }

    /**
     * Cleans up old metrics based on retention period
     */
    private cleanupOldMetrics(): void {
        const cutoffTime = Date.now() - this.retentionPeriodMs;
        for (const [name, metricsList] of this.metrics.entries()) {
            this.metrics.set(
                name,
                metricsList.filter(m => m.timestamp >= cutoffTime)
            );
        }
    }

    /**
     * Cleanup resources
     */
    public dispose(): void {
        clearInterval(this.cleanupInterval);
        this.metrics.clear();
    }
}

// Export singleton instance with default configuration
export const metricsCollector = new MetricsCollector({
    retentionPeriodMs: 3600000, // 1 hour
    thresholds: {
        maxInferenceTime: PERFORMANCE_THRESHOLDS.MAX_INFERENCE_TIME,
        maxMemoryUsage: MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE,
        maxGPUMemory: MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE,
        targetFPS: PERFORMANCE_THRESHOLDS.FPS_TARGET
    },
    enableAlerts: true
});

export default metricsCollector;