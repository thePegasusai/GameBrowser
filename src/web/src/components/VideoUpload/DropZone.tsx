/**
 * Memory-aware drag-and-drop video upload component for browser-based video game diffusion model
 * @version 1.0.0
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';
import styled from 'styled-components';
import { Alert } from '../Common/Alert';
import { useVideo } from '../../hooks/useVideo';
import { validateMemoryConstraints } from '../../lib/utils/validation';
import { PERFORMANCE_THRESHOLDS, MEMORY_CONSTRAINTS } from '../../constants/model';
import { VideoProcessingState } from '../../types/video';

// Styled components with enhanced accessibility
const DropZoneContainer = styled.div<{ isDragActive: boolean; isProcessing: boolean }>`
  width: 100%;
  height: 200px;
  border: 2px dashed ${props => 
    props.isProcessing ? props.theme.palette.primary.main :
    props.isDragActive ? props.theme.palette.secondary.main :
    props.theme.palette.grey[300]};
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background-color: ${props =>
    props.isDragActive ? props.theme.palette.action.hover :
    props.isProcessing ? props.theme.palette.action.selected :
    props.theme.palette.background.paper};
  transition: all 0.3s ease-in-out;
  cursor: ${props => props.isProcessing ? 'wait' : 'pointer'};
  position: relative;
  outline: none;

  &:focus-visible {
    box-shadow: 0 0 0 2px ${props => props.theme.palette.primary.main};
  }

  &:hover {
    background-color: ${props =>
      props.isProcessing ? props.theme.palette.action.selected :
      props.theme.palette.action.hover};
  }
`;

const UploadIcon = styled.div`
  font-size: 48px;
  color: ${props => props.theme.palette.text.secondary};
  margin-bottom: 16px;
`;

const UploadText = styled.p`
  color: ${props => props.theme.palette.text.primary};
  margin: 8px 0;
  text-align: center;
`;

const HiddenInput = styled.input`
  display: none;
`;

// Interface definitions
interface DropZoneProps {
  onFileAccepted: (file: File) => Promise<void>;
  onError: (error: UploadError) => void;
  maxSize?: number;
  accept?: string[];
  className?: string;
  ariaLabel?: string;
  processingTimeout?: number;
}

interface UploadError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

const DEFAULT_MAX_SIZE = 100 * 1024 * 1024; // 100MB
const DEFAULT_ACCEPT = ['video/mp4', 'video/webm'];
const MEMORY_CHECK_INTERVAL = 5000; // 5 seconds

/**
 * Memory-aware dropzone component for video uploads
 */
const DropZone: React.FC<DropZoneProps> = React.memo(({
  onFileAccepted,
  onError,
  maxSize = DEFAULT_MAX_SIZE,
  accept = DEFAULT_ACCEPT,
  className,
  ariaLabel = 'Drop video file here or click to upload',
  processingTimeout = 30000
}) => {
  // State management
  const [isDragActive, setIsDragActive] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<UploadError | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { processVideo, state, cleanup } = useVideo();

  // Memory monitoring
  useEffect(() => {
    let memoryCheckInterval: number;

    if (isProcessing) {
      memoryCheckInterval = window.setInterval(async () => {
        try {
          const memoryResult = await validateMemoryConstraints(
            MEMORY_CONSTRAINTS.MAX_GPU_MEMORY_USAGE,
            { gpuMemory: MEMORY_CONSTRAINTS.MAX_TENSOR_BUFFER_SIZE }
          );

          if (!memoryResult.canAllocate) {
            handleError({
              code: 'MEMORY_EXCEEDED',
              message: 'Memory limit exceeded during processing',
              details: { memoryInfo: memoryResult }
            });
            await cleanup();
          }
        } catch (error) {
          console.error('Memory check failed:', error);
        }
      }, MEMORY_CHECK_INTERVAL);
    }

    return () => {
      if (memoryCheckInterval) {
        window.clearInterval(memoryCheckInterval);
      }
    };
  }, [isProcessing, cleanup]);

  // File validation with memory check
  const validateFile = useCallback(async (file: File): Promise<boolean> => {
    if (!file) return false;

    try {
      // Check file type
      if (!accept.includes(file.type)) {
        throw new Error(`Unsupported file type: ${file.type}`);
      }

      // Check file size
      if (file.size > maxSize) {
        throw new Error(`File size exceeds ${maxSize / (1024 * 1024)}MB limit`);
      }

      // Check memory availability
      const memoryResult = await validateMemoryConstraints(
        file.size,
        { gpuMemory: file.size * 1.5 } // Estimate GPU memory needed
      );

      if (!memoryResult.canAllocate) {
        throw new Error('Insufficient memory available for processing');
      }

      return true;
    } catch (error) {
      handleError({
        code: 'VALIDATION_FAILED',
        message: error.message,
        details: { fileName: file.name, fileSize: file.size }
      });
      return false;
    }
  }, [accept, maxSize]);

  // File processing handler
  const handleFile = useCallback(async (file: File) => {
    if (!await validateFile(file)) return;

    try {
      setIsProcessing(true);
      setError(null);

      const processingTimeout = setTimeout(() => {
        handleError({
          code: 'PROCESSING_TIMEOUT',
          message: 'Video processing timed out',
          details: { fileName: file.name }
        });
      }, processingTimeout);

      await onFileAccepted(file);
      clearTimeout(processingTimeout);

    } catch (error) {
      handleError({
        code: 'PROCESSING_ERROR',
        message: error.message,
        details: { fileName: file.name }
      });
    } finally {
      setIsProcessing(false);
    }
  }, [validateFile, onFileAccepted, processingTimeout]);

  // Error handler
  const handleError = useCallback((error: UploadError) => {
    setError(error);
    onError(error);
    setIsProcessing(false);
  }, [onError]);

  // Drag event handlers
  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    const file = e.dataTransfer.files[0];
    if (file) {
      await handleFile(file);
    }
  }, [handleFile]);

  // Click handler
  const handleClick = useCallback(() => {
    if (!isProcessing && fileInputRef.current) {
      fileInputRef.current.click();
    }
  }, [isProcessing]);

  return (
    <>
      <DropZoneContainer
        className={className}
        isDragActive={isDragActive}
        isProcessing={isProcessing}
        onDragEnter={handleDragEnter}
        onDragOver={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
        role="button"
        tabIndex={0}
        aria-label={ariaLabel}
      >
        <UploadIcon>
          {isProcessing ? '⏳' : '📁'}
        </UploadIcon>
        <UploadText>
          {isProcessing ? 'Processing video...' :
           isDragActive ? 'Drop video here' :
           'Drag and drop video or click to upload'}
        </UploadText>
        {!isProcessing && (
          <UploadText>
            Supported formats: {accept.map(type => type.split('/')[1]).join(', ')}
          </UploadText>
        )}
        <HiddenInput
          ref={fileInputRef}
          type="file"
          accept={accept.join(',')}
          onChange={e => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
          aria-hidden="true"
        />
      </DropZoneContainer>
      
      {error && (
        <Alert 
          severity="error"
          onClose={() => setError(null)}
          dismissible
        >
          {error.message}
        </Alert>
      )}
    </>
  );
});

DropZone.displayName = 'DropZone';

export default DropZone;