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
                  let paths = dataIndex[0].path;
                   let imagePaths = paths.slice(0, 1);
                   let actions = dataIndex[0].action;
                 let frames = [];
                let image;
               for(let path of imagePaths){
                image = await loadVideoFromPath(path);
                 frames.push(image)
               }
                let tensor = tf.concat(frames, 0)
                 await train(dataIndex, tensor)
               let generated = await generateVideo(tensor, [actions])
                let canvas
                for(let frame of tf.unstack(generated[0], 0)){
                    canvas = document.createElement('canvas');
                    canvas.width = frame.shape[1]
                    canvas.height = frame.shape[0]
                    generationContainer.appendChild(canvas)
                    await tf.browser.toPixels(tf.clipByValue(frame, 0, 1), canvas)
                 }
             }
            catch(e){
               console.error('Error during training', e);
                alert('An error occurred during training, please check the console')
           }
        })
    </script>
</body>
</html>