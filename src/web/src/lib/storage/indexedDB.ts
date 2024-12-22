/**
 * @fileoverview Enhanced IndexedDB storage implementation for browser-based video game diffusion model
 * @version 1.0.0
 * @license MIT
 */

import { Tensor } from '@tensorflow/tfjs-core'; // v4.x
import { compress } from 'lz-string'; // v1.5.0
import { INDEXED_DB_CONFIG } from '../../config/storage';
import { Logger } from '../../utils/logger';

/**
 * Custom error type for storage operations with detailed tracking
 */
export interface StorageError extends Error {
  code: string;
  message: string;
  details: any;
  browserInfo: string;
  timestamp: number;
}

/**
 * Configuration options for storage operations
 */
export interface StorageOptions {
  compression?: boolean;
  chunkSize?: number;
  retryAttempts?: number;
  validateData?: boolean;
}

/**
 * Enhanced IndexedDB storage manager with memory tracking and error handling
 */
export class IndexedDBStorage {
  private db: IDBDatabase | null = null;
  private readonly logger: Logger;
  private readonly pendingTransactions: Map<string, Promise<void>>;
  private readonly metrics: Map<string, number>;
  private readonly options: Required<StorageOptions>;

  constructor(options: StorageOptions = {}) {
    this.logger = new Logger({
      level: 'info',
      namespace: 'IndexedDBStorage',
      persistLogs: true
    });

    this.pendingTransactions = new Map();
    this.metrics = new Map();
    
    // Set default options
    this.options = {
      compression: true,
      chunkSize: 16 * 1024 * 1024, // 16MB chunks
      retryAttempts: 3,
      validateData: true,
      ...options
    };

    // Initialize metrics tracking
    this.initializeMetrics();
  }

  /**
   * Initializes database connection with version management
   */
  public async initialize(): Promise<void> {
    try {
      // Check browser compatibility
      if (!this.checkBrowserSupport()) {
        throw new Error('IndexedDB not supported in this browser');
      }

      // Request storage quota if needed
      await this.requestStorageQuota();

      // Open database connection
      const request = indexedDB.open(INDEXED_DB_CONFIG.name, INDEXED_DB_CONFIG.version);

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        
        // Create object stores if they don't exist
        if (!db.objectStoreNames.contains(INDEXED_DB_CONFIG.stores.MODEL_WEIGHTS)) {
          db.createObjectStore(INDEXED_DB_CONFIG.stores.MODEL_WEIGHTS, { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains(INDEXED_DB_CONFIG.stores.CHECKPOINTS)) {
          db.createObjectStore(INDEXED_DB_CONFIG.stores.CHECKPOINTS, { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains(INDEXED_DB_CONFIG.stores.TRAINING_DATA)) {
          db.createObjectStore(INDEXED_DB_CONFIG.stores.TRAINING_DATA, { keyPath: 'id' });
        }
      };

      this.db = await new Promise((resolve, reject) => {
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      });

      this.logger.log('IndexedDB initialized successfully');
    } catch (error) {
      this.handleError('INIT_ERROR', 'Failed to initialize IndexedDB', error);
      throw error;
    }
  }

  /**
   * Saves model weights with compression and integrity checks
   */
  public async saveModelWeights(
    id: string,
    weights: Tensor[],
    options: StorageOptions = {}
  ): Promise<void> {
    const mergedOptions = { ...this.options, ...options };
    const transactionId = `save_${id}_${Date.now()}`;

    try {
      // Check available space
      await this.checkStorageQuota();

      // Start transaction
      const transaction = this.startTransaction(
        INDEXED_DB_CONFIG.stores.MODEL_WEIGHTS,
        'readwrite',
        transactionId
      );

      // Prepare weights data
      const weightsData = await this.prepareWeightsData(weights, mergedOptions);

      // Store weights in chunks
      await this.storeDataInChunks(
        transaction,
        id,
        weightsData,
        mergedOptions.chunkSize
      );

      // Verify stored data
      if (mergedOptions.validateData) {
        await this.verifyStoredData(id, weightsData);
      }

      // Update metrics
      this.updateMetrics('modelWeightsSaved', weightsData.byteLength);

      this.logger.log(`Model weights saved successfully: ${id}`);
    } catch (error) {
      this.handleError('SAVE_ERROR', `Failed to save model weights: ${id}`, error);
      throw error;
    } finally {
      this.pendingTransactions.delete(transactionId);
    }
  }

  /**
   * Loads model weights with validation and decompression
   */
  public async loadModelWeights(
    id: string,
    options: StorageOptions = {}
  ): Promise<Tensor[]> {
    const mergedOptions = { ...this.options, ...options };
    const transactionId = `load_${id}_${Date.now()}`;

    try {
      // Start transaction
      const transaction = this.startTransaction(
        INDEXED_DB_CONFIG.stores.MODEL_WEIGHTS,
        'readonly',
        transactionId
      );

      // Load data chunks
      const chunksData = await this.loadDataChunks(transaction, id);

      // Reconstruct and validate data
      const weightsData = await this.reconstructWeightsData(
        chunksData,
        mergedOptions
      );

      // Update metrics
      this.updateMetrics('modelWeightsLoaded', weightsData.byteLength);

      return this.deserializeWeights(weightsData);
    } catch (error) {
      this.handleError('LOAD_ERROR', `Failed to load model weights: ${id}`, error);
      throw error;
    } finally {
      this.pendingTransactions.delete(transactionId);
    }
  }

  /**
   * Clears store with safety checks and cleanup
   */
  public async clearStore(
    storeName: string,
    options: StorageOptions = {}
  ): Promise<void> {
    const transactionId = `clear_${storeName}_${Date.now()}`;

    try {
      // Validate store existence
      if (!this.db?.objectStoreNames.contains(storeName)) {
        throw new Error(`Store not found: ${storeName}`);
      }

      // Check pending transactions
      await this.waitForPendingTransactions(storeName);

      // Start clear transaction
      const transaction = this.startTransaction(storeName, 'readwrite', transactionId);
      const store = transaction.objectStore(storeName);

      await new Promise<void>((resolve, reject) => {
        const request = store.clear();
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
      });

      // Update metrics
      this.updateMetrics(`${storeName}Cleared`, 1);

      this.logger.log(`Store cleared successfully: ${storeName}`);
    } catch (error) {
      this.handleError('CLEAR_ERROR', `Failed to clear store: ${storeName}`, error);
      throw error;
    } finally {
      this.pendingTransactions.delete(transactionId);
    }
  }

  /**
   * Returns current storage metrics
   */
  public getMetrics(): Record<string, number> {
    return Object.fromEntries(this.metrics);
  }

  // Private helper methods...
  private checkBrowserSupport(): boolean {
    return 'indexedDB' in window;
  }

  private async requestStorageQuota(): Promise<void> {
    if ('storage' in navigator && 'persist' in navigator.storage) {
      await navigator.storage.persist();
    }
  }

  private async checkStorageQuota(): Promise<void> {
    if ('storage' in navigator && 'estimate' in navigator.storage) {
      const estimate = await navigator.storage.estimate();
      const usageRatio = (estimate.usage || 0) / (estimate.quota || 0);
      
      if (usageRatio > 0.9) {
        throw new Error('Storage quota exceeded');
      }
    }
  }

  private startTransaction(
    storeName: string,
    mode: IDBTransactionMode,
    transactionId: string
  ): IDBTransaction {
    if (!this.db) {
      throw new Error('Database not initialized');
    }

    const transaction = this.db.transaction(storeName, mode);
    const promise = new Promise<void>((resolve, reject) => {
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
    });

    this.pendingTransactions.set(transactionId, promise);
    return transaction;
  }

  private async waitForPendingTransactions(storeName: string): Promise<void> {
    const pending = Array.from(this.pendingTransactions.values());
    await Promise.all(pending);
  }

  private initializeMetrics(): void {
    this.metrics.set('modelWeightsSaved', 0);
    this.metrics.set('modelWeightsLoaded', 0);
    this.metrics.set('storageErrors', 0);
    this.metrics.set('compressionRatio', 0);
  }

  private updateMetrics(key: string, value: number): void {
    this.metrics.set(key, (this.metrics.get(key) || 0) + value);
  }

  private handleError(code: string, message: string, error: any): void {
    const storageError: StorageError = {
      name: 'StorageError',
      code,
      message,
      details: error,
      browserInfo: navigator.userAgent,
      timestamp: Date.now()
    };

    this.updateMetrics('storageErrors', 1);
    this.logger.error(storageError);
  }
}

export default IndexedDBStorage;