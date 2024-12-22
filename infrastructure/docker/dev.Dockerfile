# Start with Node.js 16 base image for TensorFlow.js compatibility
FROM node:16

# Build arguments
ARG NODE_VERSION=16
ARG TENSORFLOW_VERSION=4.x
ARG WEBGL_VERSION=2.0

# Set labels for container metadata
LABEL maintainer="BVGDM Team" \
      environment="development" \
      project="browser-video-game-diffusion-model" \
      version="1.0.0"

# Set environment variables
ENV NODE_ENV=development \
    PORT=3000 \
    NODE_OPTIONS="--max-old-space-size=4096" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    WEBPACK_DEV_SERVER_HOST=0.0.0.0

# Install system dependencies including WebGL support
RUN apt-get update && apt-get install -y \
    curl \
    libgl1-mesa-dev \
    libxi-dev \
    libxinerama-dev \
    libxcursor-dev \
    libwebgl1 \
    mesa-utils \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Create app directory and set permissions
WORKDIR /app
RUN chown -R node:node /app

# Switch to non-root user for security
USER node

# Copy package files with appropriate permissions
COPY --chown=node:node package*.json ./

# Install dependencies including development packages
RUN npm install && \
    # Clear npm cache for smaller image
    npm cache clean --force

# Copy source code with correct ownership
COPY --chown=node:node . .

# Set up health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Expose development server port
EXPOSE 3000

# Configure volume mount points
VOLUME ["/app", "/app/node_modules", "/root/.npm"]

# Start development server with hot reload
CMD ["npm", "run", "start"]

# Resource limits are set at runtime via docker-compose or kubernetes
# Example: docker run --memory="4g" --cpus="2" ...

# Security scanning label for CI/CD
LABEL security.scan-date="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"