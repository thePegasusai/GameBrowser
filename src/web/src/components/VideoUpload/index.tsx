/**
 * Main container component for video upload functionality with memory-aware processing
 * Implements WebGL acceleration and comprehensive error handling
 * @version 1.0.0
 */

import React, { useState, useCallback, useEffect } from 'react';
import { styled } from '@mui/material/styles';
import Box from '@mui/material/Box';
import LinearProgress from '@mui/material/LinearProgress';
import DropZone from './DropZone';
import VideoPreview from './VideoPreview';
import { useVideo } from '../../hooks/useVideo';
import { VideoFrame } from '../../types/video';

// Styled components
const UploadContainer = styled(Box)(({ theme }) => ({
  width: '100%',
  maxWidth: '800px',
  margin: '0 auto',
  padding: theme.spacing(3),
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(3),
  position: 'relative'
}));

const MemoryIndicator = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: theme.spacing(1),
  right: theme.spacing(1),
  padding: theme.spacing(0.5, 1),
  borderRadius: theme.shape.borderRadius,
  backgroundColor: theme.palette.background.paper,
  boxShadow: theme.shadows[1],
  fontSize: '0.75rem',
  color: theme.palette.text.secondary
}));

// Props interface
interface VideoUploadProps {
  onUploadComplete: (frames: VideoFrame[]) => void;
  onError: (error: Error) => void;
  maxFileSize?: number;
  acceptedFormats?: string[];
  className?: string;
  memoryThreshold?: number;
  batchSize?: number;
  webGLOptions?: WebGLContextOptions;
}

// Default WebGL context options
const DEFAULT_WEBGL_OPTIONS: WebGLContextOptions = {
  powerPreference: 'high-performance',
  failIfMajorPerformanceCaveat: true,
  desynchronized: false
};

/**
 * Memory-aware video upload component with WebGL acceleration
 */
const VideoUpload: React.FC<VideoUploadProps> = React.memo(({
  onUploadComplete,
  onError,
  maxFileSize = 100 * 1024 * 1024, // 100MB
  acceptedFormats = ['video/mp4', 'video/webm'],
  className,
  memoryThreshold = 3.5 * 1024 * 1024 * 1024, // 3.5GB
  batchSize = 4,
  webGLOptions = DEFAULT_WEBGL_OPTIONS
}) => {
  // State management
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [memoryStats, setMemoryStats] = useState({
    usage: 0,
    limit: memoryThreshold,
    pressure: false
  });

  // Initialize video processing hook
  const {
    processVideo,
    state,
    frames,
    error: processingError,
    memoryUsage,
    reset
  } = useVideo({
    batchSize,
    memoryLimit: memoryThreshold,
    webGLOptions
  });

  // Handle memory pressure
  const handleMemoryPressure = useCallback((stats: { usage: number }) => {
    setMemoryStats(prev => ({
      ...prev,
      usage: stats.usage,
      pressure: stats.usage > memoryThreshold * 0.9
    }));

    if (stats.usage > memoryThreshold * 0.9) {
      onError(new Error('Memory usage approaching limit'));
    }
  }, [memoryThreshold, onError]);

  // Handle file acceptance
  const handleFileAccepted = useCallback(async (file: File) => {
    try {
      setSelectedFile(file);
      await processVideo(file);
    } catch (err) {
      onError(err instanceof Error ? err : new Error('File processing failed'));
      reset();
    }
  }, [processVideo, onError, reset]);

  // Handle processing completion
  const handleProcessingComplete = useCallback((
    processedFrames: VideoFrame[],
    stats: { totalUsed: number }
  ) => {
    setMemoryStats(prev => ({
      ...prev,
      usage: stats.totalUsed
    }));
    onUploadComplete(processedFrames);
  }, [onUploadComplete]);

  // Handle WebGL context errors
  useEffect(() => {
    const handleContextLoss = (event: WebGLContextEvent) => {
      event.preventDefault();
      onError(new Error('WebGL context lost - please try again'));
      reset();
    };

    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    
    if (gl) {
      canvas.addEventListener('webglcontextlost', handleContextLoss);
      return () => {
        canvas.removeEventListener('webglcontextlost', handleContextLoss);
      };
    }
  }, [onError, reset]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      reset();
      if (selectedFile) {
        setSelectedFile(null);
      }
    };
  }, [reset]);

  return (
    <UploadContainer className={className}>
      <DropZone
        onFileAccepted={handleFileAccepted}
        onError={onError}
        maxFileSize={maxFileSize}
        acceptedFormats={acceptedFormats}
        disabled={state === 'processing'}
      />

      {selectedFile && (
        <VideoPreview
          file={selectedFile}
          onProcessingComplete={handleProcessingComplete}
          onMemoryPressure={handleMemoryPressure}
        />
      )}

      {state === 'processing' && (
        <LinearProgress 
          variant="determinate" 
          value={(frames.length / (selectedFile?.size || 1)) * 100}
        />
      )}

      <MemoryIndicator>
        Memory: {Math.round(memoryStats.usage / (1024 * 1024))}MB / 
        {Math.round(memoryThreshold / (1024 * 1024))}MB
        {memoryStats.pressure && ' (High Usage)'}
      </MemoryIndicator>
    </UploadContainer>
  );
});

VideoUpload.displayName = 'VideoUpload';

export default VideoUpload;