import path from 'path';
import webpack from 'webpack';  // v5.80.0
import TerserPlugin from 'terser-webpack-plugin';  // v5.3.0
import WebpackDevServer from 'webpack-dev-server';  // v4.13.0
import CompressionPlugin from 'compression-webpack-plugin';  // v10.0.0
import { BundleAnalyzerPlugin } from 'webpack-bundle-analyzer';  // v4.9.0

interface WebpackEnv {
  production: boolean;
  analyze?: boolean;
}

function getWebpackConfig(env: WebpackEnv): webpack.Configuration {
  const isProduction = env.production;
  const shouldAnalyze = env.analyze;

  const config: webpack.Configuration = {
    mode: isProduction ? 'production' : 'development',
    devtool: isProduction ? 'source-map' : 'eval-source-map',
    
    entry: {
      main: './src/index.tsx',
      'generation.worker': './src/workers/generation.worker.ts',
      'training.worker': './src/workers/training.worker.ts',
      'video.worker': './src/workers/video.worker.ts',
    },

    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: isProduction ? '[name].[contenthash].js' : '[name].js',
      chunkFilename: isProduction ? '[name].[contenthash].chunk.js' : '[name].chunk.js',
      publicPath: '/',
      globalObject: 'self',
      clean: true,
    },

    module: {
      rules: [
        {
          test: /\.tsx?$/,
          exclude: /node_modules/,
          use: {
            loader: 'babel-loader',
            options: {
              presets: [
                '@babel/preset-env',
                '@babel/preset-typescript',
                '@babel/preset-react'
              ],
              plugins: [
                '@babel/plugin-transform-runtime',
                isProduction && 'transform-remove-console'
              ].filter(Boolean),
            },
          },
        },
        {
          test: /\.worker\.ts$/,
          use: {
            loader: 'worker-loader',
            options: {
              filename: isProduction ? '[name].[contenthash].worker.js' : '[name].worker.js',
              worker: {
                type: 'module',
              },
            },
          },
        },
        {
          test: /\.css$/,
          use: ['style-loader', 'css-loader'],
        },
        {
          test: /\.(png|svg|jpg|jpeg|gif|ico)$/,
          type: 'asset/resource',
          generator: {
            filename: 'assets/[name].[hash][ext]',
          },
        },
      ],
    },

    resolve: {
      extensions: ['.tsx', '.ts', '.js'],
      alias: {
        '@': path.resolve(__dirname, 'src'),
        '@components': path.resolve(__dirname, 'src/components'),
        '@lib': path.resolve(__dirname, 'src/lib'),
        '@hooks': path.resolve(__dirname, 'src/hooks'),
        '@workers': path.resolve(__dirname, 'src/workers'),
        '@config': path.resolve(__dirname, 'src/config'),
        '@constants': path.resolve(__dirname, 'src/constants'),
        '@types': path.resolve(__dirname, 'src/types'),
        '@styles': path.resolve(__dirname, 'src/styles'),
      },
    },

    optimization: {
      minimize: isProduction,
      minimizer: [
        new TerserPlugin({
          terserOptions: {
            compress: {
              drop_console: isProduction,
              passes: 2,
            },
            mangle: {
              // Preserve TensorFlow.js class names
              keep_classnames: /^tf\./,
              keep_fnames: /^tf\./,
            },
          },
        }),
      ],
      splitChunks: {
        chunks: 'all',
        maxInitialRequests: 25,
        maxAsyncRequests: 25,
        cacheGroups: {
          tensorflow: {
            test: /[\\/]node_modules[\\/]@tensorflow/,
            name: 'tensorflow',
            chunks: 'all',
            priority: 10,
            enforce: true,
            reuseExistingChunk: true,
          },
          webgl: {
            test: /[\\/]node_modules[\\/].*webgl/,
            name: 'webgl',
            chunks: 'all',
            priority: 9,
            reuseExistingChunk: true,
          },
          workers: {
            test: /[\\/]src[\\/]workers/,
            name: 'workers',
            chunks: 'all',
            priority: 8,
            reuseExistingChunk: true,
          },
          vendors: {
            test: /[\\/]node_modules[\\/]/,
            name: 'vendors',
            chunks: 'all',
            priority: 7,
            reuseExistingChunk: true,
          },
        },
      },
      runtimeChunk: 'single',
    },

    plugins: [
      new webpack.DefinePlugin({
        'process.env.NODE_ENV': JSON.stringify(isProduction ? 'production' : 'development'),
        'process.env.TENSORFLOW_WEBGL_VERSION': JSON.stringify(2),
        'process.env.TENSORFLOW_WEBGL_CPU_FORWARD': JSON.stringify(false),
      }),
      // Enable WebGL memory management
      new webpack.ProvidePlugin({
        tf: '@tensorflow/tfjs',
      }),
      isProduction && new CompressionPlugin({
        test: /\.(js|css|html|svg)$/,
        algorithm: 'gzip',
        threshold: 10240,
        minRatio: 0.8,
      }),
      shouldAnalyze && new BundleAnalyzerPlugin({
        analyzerMode: 'static',
        reportFilename: 'bundle-analysis.html',
      }),
    ].filter(Boolean),

    devServer: {
      port: 3000,
      hot: true,
      historyApiFallback: true,
      static: {
        directory: path.join(__dirname, 'public'),
      },
      headers: {
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
      },
      client: {
        webSocketURL: {
          hostname: 'localhost',
        },
      },
      // WebGL context preservation
      setupMiddlewares: (middlewares, devServer) => {
        if (!devServer) {
          throw new Error('webpack-dev-server is not defined');
        }
        return middlewares;
      },
    },

    performance: {
      maxEntrypointSize: 512000,
      maxAssetSize: 512000,
      hints: isProduction ? 'warning' : false,
    },
  };

  return config;
}

export default getWebpackConfig;