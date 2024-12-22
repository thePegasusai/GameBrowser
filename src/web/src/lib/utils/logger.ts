/**
 * @fileoverview Enhanced logging utility for browser-based video game diffusion model
 * @version 1.0.0
 * @license MIT
 */

import debug from 'debug'; // v4.x
import { validateMemoryConstraints } from './validation';

/**
 * Log levels with color coding for console output
 */
export enum LogLevel {
  DEBUG = 'debug',
  INFO = 'info',
  WARN = 'warn',
  ERROR = 'error'
}

/**
 * Interface for structured log context
 */
interface LogContext {
  component?: string;
  operation?: string;
  timestamp?: number;
  memoryInfo?: MemoryInfo;
  browserInfo?: BrowserInfo;
  [key: string]: any;
}

/**
 * Interface for memory usage tracking
 */
interface MemoryInfo {
  heapUsed: number;
  heapTotal: number;
  gpuMemory?: number;
  tensorCount?: number;
}

/**
 * Interface for browser-specific context
 */
interface BrowserInfo {
  userAgent: string;
  platform: string;
  webglVersion: string;
  language: string;
}

/**
 * Interface for performance metrics
 */
interface PerformanceMetric {
  name: string;
  value: number;
  timestamp: number;
  type: 'memory' | 'performance' | 'gpu';
}

/**
 * Interface for error tracking
 */
interface ErrorMetric {
  message: string;
  stack?: string;
  count: number;
  firstSeen: number;
  lastSeen: number;
  context?: LogContext;
}

/**
 * Configuration for log batching
 */
interface BatchConfig {
  enabled: boolean;
  maxSize: number;
  flushInterval: number;
}

/**
 * Logger configuration interface
 */
interface LoggerConfig {
  level: LogLevel;
  persistLogs?: boolean;
  metricsRetentionMs?: number;
  memoryThreshold?: number;
  namespace?: string;
  batch?: BatchConfig;
}

/**
 * Enhanced logging class with memory-aware constraints and performance monitoring
 */
export class Logger {
  private readonly debugLogger: debug.Debugger;
  private level: LogLevel;
  private persistLogs: boolean;
  private metrics: Map<string, PerformanceMetric>;
  private errorMetrics: Map<string, ErrorMetric>;
  private metricsRetentionMs: number;
  private memoryThreshold: number;
  private batchConfig: BatchConfig;
  private logBatch: string[];
  private batchTimeout?: number;

  constructor(config: LoggerConfig) {
    this.level = config.level;
    this.persistLogs = config.persistLogs || false;
    this.metricsRetentionMs = config.metricsRetentionMs || 3600000; // 1 hour
    this.memoryThreshold = config.memoryThreshold || 0.9; // 90%
    this.metrics = new Map();
    this.errorMetrics = new Map();
    this.debugLogger = debug(config.namespace || 'bvgdm');
    this.batchConfig = config.batch || { enabled: false, maxSize: 100, flushInterval: 5000 };
    this.logBatch = [];

    // Initialize batch processing if enabled
    if (this.batchConfig.enabled) {
      this.initializeBatchProcessing();
    }

    // Schedule periodic cleanup
    this.scheduleMetricsCleanup();
  }

  /**
   * Memory-aware logging with batching support
   */
  public async log(message: string, level: LogLevel = LogLevel.INFO, context: LogContext = {}): Promise<void> {
    try {
      // Check memory constraints
      const memoryInfo = await this.getMemoryInfo();
      const memoryResult = await validateMemoryConstraints(
        memoryInfo.heapUsed,
        { gpuMemory: memoryInfo.gpuMemory }
      );

      if (!memoryResult.canAllocate && level !== LogLevel.ERROR) {
        this.handleMemoryConstraint(message, level, context);
        return;
      }

      // Format log message
      const formattedMessage = this.formatLogMessage(message, level, {
        ...context,
        memoryInfo,
        browserInfo: this.getBrowserInfo()
      });

      // Handle batching
      if (this.batchConfig.enabled && level !== LogLevel.ERROR) {
        this.addToBatch(formattedMessage);
      } else {
        this.writeLog(formattedMessage, level);
      }

      // Update metrics
      this.recordMetric('log_count', 1, 'performance', { level });

    } catch (error) {
      console.error('Logging error:', error);
      this.handleError(error, { originalMessage: message, level, context });
    }
  }

  /**
   * Enhanced error logging with detailed context
   */
  public async error(error: Error, context: LogContext = {}): Promise<void> {
    const errorContext = {
      ...context,
      stack: error.stack,
      browserInfo: this.getBrowserInfo(),
      memoryInfo: await this.getMemoryInfo()
    };

    // Track error metrics
    this.trackError(error, errorContext);

    // Force immediate logging for errors
    await this.log(error.message, LogLevel.ERROR, errorContext);

    // Record error metric
    this.recordMetric('error_count', 1, 'performance', { 
      errorType: error.name,
      component: context.component
    });
  }

  /**
   * Records performance metrics with categorization
   */
  public recordMetric(
    name: string,
    value: number,
    type: 'memory' | 'performance' | 'gpu',
    context: Record<string, any> = {}
  ): void {
    const metric: PerformanceMetric = {
      name,
      value,
      timestamp: Date.now(),
      type
    };

    this.metrics.set(`${name}_${Date.now()}`, metric);

    // Log metric in debug mode
    this.debugLogger(`Metric [${type}] ${name}: ${value}`);

    // Clean up old metrics if needed
    this.cleanupMetrics();
  }

  /**
   * Formats log message with enhanced context
   */
  private formatLogMessage(message: string, level: LogLevel, context: LogContext): string {
    const timestamp = new Date().toISOString();
    const memoryInfo = context.memoryInfo ? 
      `[Memory: ${Math.round(context.memoryInfo.heapUsed / 1024 / 1024)}MB]` : '';
    
    return JSON.stringify({
      timestamp,
      level,
      message,
      ...context,
      formattedMessage: `${timestamp} ${level.toUpperCase()} ${memoryInfo} ${message}`
    });
  }

  /**
   * Writes log to appropriate output
   */
  private writeLog(message: string, level: LogLevel): void {
    const consoleMethod = level === LogLevel.ERROR ? 'error' : 
                         level === LogLevel.WARN ? 'warn' : 
                         level === LogLevel.DEBUG ? 'debug' : 'log';
    
    console[consoleMethod](message);

    if (this.persistLogs) {
      this.persistLog(message);
    }
  }

  /**
   * Persists log to browser storage
   */
  private persistLog(message: string): void {
    try {
      const logs = JSON.parse(localStorage.getItem('bvgdm_logs') || '[]');
      logs.push(message);
      localStorage.setItem('bvgdm_logs', JSON.stringify(logs.slice(-1000))); // Keep last 1000 logs
    } catch (error) {
      console.error('Error persisting log:', error);
    }
  }

  /**
   * Initializes batch processing
   */
  private initializeBatchProcessing(): void {
    this.batchTimeout = window.setInterval(() => {
      this.flushBatch();
    }, this.batchConfig.flushInterval);
  }

  /**
   * Adds log message to batch
   */
  private addToBatch(message: string): void {
    this.logBatch.push(message);
    if (this.logBatch.length >= this.batchConfig.maxSize) {
      this.flushBatch();
    }
  }

  /**
   * Flushes log batch
   */
  private flushBatch(): void {
    if (this.logBatch.length === 0) return;

    const batch = this.logBatch.slice();
    this.logBatch = [];

    batch.forEach(message => {
      this.writeLog(message, LogLevel.INFO);
    });
  }

  /**
   * Handles memory constraint violations
   */
  private handleMemoryConstraint(message: string, level: LogLevel, context: LogContext): void {
    const warningMessage = 'Memory threshold exceeded, dropping non-critical log';
    console.warn(warningMessage, { message, level, context });
    this.recordMetric('memory_constraint_hits', 1, 'memory');
  }

  /**
   * Tracks error occurrences
   */
  private trackError(error: Error, context: LogContext): void {
    const errorKey = `${error.name}:${error.message}`;
    const existing = this.errorMetrics.get(errorKey);

    if (existing) {
      existing.count++;
      existing.lastSeen = Date.now();
      existing.context = context;
    } else {
      this.errorMetrics.set(errorKey, {
        message: error.message,
        stack: error.stack,
        count: 1,
        firstSeen: Date.now(),
        lastSeen: Date.now(),
        context
      });
    }
  }

  /**
   * Gets current memory information
   */
  private async getMemoryInfo(): Promise<MemoryInfo> {
    const memory = performance?.memory || { usedJSHeapSize: 0, totalJSHeapSize: 0 };
    return {
      heapUsed: memory.usedJSHeapSize,
      heapTotal: memory.totalJSHeapSize,
      gpuMemory: await this.getGPUMemory()
    };
  }

  /**
   * Gets browser information
   */
  private getBrowserInfo(): BrowserInfo {
    return {
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      webglVersion: this.getWebGLVersion(),
      language: navigator.language
    };
  }

  /**
   * Gets WebGL version information
   */
  private getWebGLVersion(): string {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    return gl ? gl.getParameter(gl.VERSION) : 'Not available';
  }

  /**
   * Gets GPU memory usage if available
   */
  private async getGPUMemory(): Promise<number | undefined> {
    try {
      if ('gpu' in navigator) {
        const adapter = await (navigator as any).gpu.requestAdapter();
        return adapter?.limits?.maxBufferSize;
      }
    } catch {
      return undefined;
    }
  }

  /**
   * Schedules periodic metrics cleanup
   */
  private scheduleMetricsCleanup(): void {
    setInterval(() => {
      this.cleanupMetrics();
    }, 60000); // Clean up every minute
  }

  /**
   * Cleans up old metrics
   */
  private cleanupMetrics(): void {
    const now = Date.now();
    for (const [key, metric] of this.metrics.entries()) {
      if (now - metric.timestamp > this.metricsRetentionMs) {
        this.metrics.delete(key);
      }
    }
  }

  /**
   * Handles internal logger errors
   */
  private handleError(error: Error, context: any): void {
    console.error('Logger internal error:', error, context);
    this.trackError(error, { internal: true, ...context });
  }
}

export default Logger;