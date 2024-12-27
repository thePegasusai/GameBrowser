#!/bin/bash

# Check if the user provided a YouTube URL (we will ignore this)
if [ -z "$1" ]; then
    echo "No Youtube URL Provided, continuing."
fi

YOUTUBE_URL="$1"
OUTPUT_DIR="sample_data"
# Extract only the video ID part from the URL, we no longer use this
VIDEO_ID=$(echo "$YOUTUBE_URL" | sed 's/.*\/\(.*\)\(.*\)/\1/' | sed 's/[^a-zA-Z0-9_-]//g')
VIDEO_PATH="${OUTPUT_DIR}/${VIDEO_ID}.mp4"
FRAMES_DIR="${OUTPUT_DIR}/frames"
ACTION_FILE="${OUTPUT_DIR}/actions.json"
DATA_INDEX="${OUTPUT_DIR}/data_index.json"
HTML_FILE="index.html"

echo "Starting setup process..."

# 1. Ensure the output directory exists
mkdir -p "${OUTPUT_DIR}"

# 2. Skip Video Download and Check
echo "Skipping video download. Please upload a video in the browser."

# 3. Skip Frame Extraction
echo "Skipping frame extraction, as this is now done on the client side."

# 4. Create Dummy Action File (Replace this with real labeling or action extraction as necessary)
echo "Creating dummy action file: ${ACTION_FILE}"

echo '{"frame_0001.png": {"movement": [0, 0, 0, 0], "camera": [0, 0], "interaction": 0}}' > $ACTION_FILE

# 5. Create Data Index (Modify paths and action file reading)
echo "Generating Data Index File: ${DATA_INDEX}"
python -c "
import os
import json

action_file = '${ACTION_FILE}'
output_file = '${DATA_INDEX}'
max_frames = 32
if not os.path.exists(action_file):
  action_data = {}
else:
  with open(action_file, 'r') as f:
    action_data = json.load(f)

data_index = []
data_index.append({
        'path': [ '' ], # dummy path, we no longer use the server side path
        'action': [action_data['frame_0001.png']] # pass along the action
    })

with open(output_file, 'w') as outfile:
    json.dump(data_index, outfile)
"

# Check if the data index was successfully created
if [ ! -f "${DATA_INDEX}" ]; then
   echo "Error: could not generate the data index."
   exit 1
fi
# 6. Create a Basic HTML UI
echo "Creating basic HTML UI: ${HTML_FILE}"

cat << EOF > "${HTML_FILE}"
<!DOCTYPE html>
<html>
<head>
    <title>Minecraft DiT Training</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
    <style>
        body { font-family: sans-serif; padding: 20px; }
         #uploadContainer{display: flex; gap: 10px; margin-bottom: 10px;}
        #videoContainer { display: flex; flex-wrap: wrap; gap: 10px; }
        video { width: 300px; height: auto; }
        #generationContainer{margin-top: 20px}
        canvas{ width: 300px; height: auto;}
        button {margin-top: 10px; background-color: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer;}
    </style>
</head>
<body>
    <h1>Minecraft DiT Training</h1>
   <div id="uploadContainer">
        <label for="videoUpload">Upload Video:</label>
        <input type="file" id="videoUpload" accept="video/*">
    </div>
        <h2>Video Input</h2>
        <div id="videoContainer"></div>
        <button id="trainButton">Start Training</button>
        <div id = "generationContainer"></div>
    <script type="module">
      import { train } from './train.js';
      import { generateVideo } from './generate.js'
      import {MinecraftDataset} from './dataset.js';
        const videoContainer = document.getElementById('videoContainer');
        const trainButton = document.getElementById("trainButton")
         const generationContainer = document.getElementById("generationContainer")
           const videoUpload = document.getElementById("videoUpload");
           let dataIndex = []

          async function loadVideoFromPath(path){
             return new Promise((resolve, reject) => {
                  const video = document.createElement('video');
                  video.src = path;
                  video.controls = true;
                  video.width = 300;
                   video.height = 200;
                  videoContainer.appendChild(video);

                 video.onloadeddata = () => {
                      let image =  tf.browser.fromPixels(video)
                     resolve(image)
                     URL.revokeObjectURL(path);
                    }
                 video.onerror = (e) => {
                        reject(e)
                    }
              })
        }

        function processActions(actions) {
          return actions
        }

         videoUpload.addEventListener("change", async (event) => {
             const file = event.target.files[0];
             const videoURL = URL.createObjectURL(file);

                let path = videoURL;

              dataIndex = [{
                  path: [path],
                   action: [{"movement": [0, 0, 0, 0], "camera": [0, 0], "interaction": 0}] // replace this to get real actions
                }];
            });


         trainButton.addEventListener("click", async() => {
             if(dataIndex.length == 0){
                alert('Please upload a video first');
                return;
              }
              try{
                 await train(dataIndex)
             }
            catch(e){
                 console.error('error during training', e);
                alert('An error occurred during training, please check the console')
            }
             let paths = dataIndex[0].path;
             let imagePaths = paths.slice(0, 1);
             let actions = dataIndex[0].action;

              let frames = [];
              for(let path of imagePaths){
               let image = await loadVideoFromPath(path);
               frames.push(image)
               }
             let tensor = tf.concat(frames, 0)
               let generated = await generateVideo(tensor, [actions])
               let canvas
             for(let frame of tf.unstack(generated[0], 0)){
                   canvas = document.createElement('canvas');
                   canvas.width = frame.shape[1]
                   canvas.height = frame.shape[0]
                   generationContainer.appendChild(canvas)
                   await tf.browser.toPixels(tf.clipByValue(frame, 0, 1), canvas)
              }
           })
    </script>
</body>
</html>
EOF

# Serve the files on a local webserver using python
echo "serving files on port 8000, please open http://localhost:8000/index.html"
python3 -m http.server 8000