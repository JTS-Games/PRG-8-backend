import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";
const enableWebcamButton = document.getElementById("webcamButton")
const logButton = document.getElementById("logButton")
const addPose1Button = document.getElementById("addPose1Button")
const addPose2Button = document.getElementById("addPose2Button")
const addPose3Button = document.getElementById("addPose3Button")
const addPose4Button = document.getElementById("addPose4Button")
const addTrainButton = document.getElementById("addTrainButton")
const addAccuracyButton = document.getElementById("addAccuracyButton")

const video = document.getElementById("webcam")
const canvasElement = document.getElementById("output_canvas")
const canvasCtx = canvasElement.getContext("2d")

const drawUtils = new DrawingUtils(canvasCtx)
let poseLandmarker = undefined;
let webcamRunning = false;
let results = undefined;

let image = document.querySelector("#myimage")

let allposes = []
let trainData = []
let testData = []

ml5.setBackend("webgl");
const nn = ml5.neuralNetwork({ task: 'classification', debug: true })

// ********************************************************************
// if webcam access, load landmarker and enable webcam button
// ********************************************************************
function startApp() {
    const hasGetUserMedia = () => { var _a; return !!((_a = navigator.mediaDevices) === null || _a === void 0 ? void 0 : _a.getUserMedia); };
    if (hasGetUserMedia()) {
        createPoseLandmarker();
    } else {
        console.warn("getUserMedia() is not supported by your browser");
    }
}

/********************************************************************
 // CREATE THE POSE DETECTOR
 ********************************************************************/
const createPoseLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 2
    });
    enableWebcamButton.addEventListener("click", enableCam);
    console.log("poselandmarker is ready!")

    enableWebcamButton.addEventListener("click", (e) => enableCam(e));
    logButton.addEventListener("click", logAllPoses);
    addPose1Button.addEventListener("click", addPose1);
    addPose2Button.addEventListener("click", addPose2);
    addPose3Button.addEventListener("click", addPose3);
    addPose4Button.addEventListener("click", addPose4);
    addTrainButton.addEventListener("click", train);
    addAccuracyButton.addEventListener("click", testAccuracy);
}

/********************************************************************
 // START THE WEBCAM
 ********************************************************************/
async function enableCam() {
    webcamRunning = true;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => {
            canvasElement.style.width = video.videoWidth;
            canvasElement.style.height = video.videoHeight;
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;
            document.querySelector(".videoView").style.height = video.videoHeight + "px";
            predictWebcam();
        });
    } catch (error) {
        console.error("Error accessing webcam:", error);
    }
}

/********************************************************************
 // START PREDICTIONS
 ********************************************************************/
async function predictWebcam() {
    results = await poseLandmarker.detectForVideo(video, performance.now())

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    for(let pose of results.landmarks){
        drawUtils.drawLandmarks(pose, { radius: 4, color: "#FF0000", lineWidth: 2 });
    }

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam)
    }
}

/********************************************************************
 // LOG POSE COORDINATES IN THE CONSOLE
 ********************************************************************/
async function logAllPoses(){
    results = await poseLandmarker.detectForVideo(video, performance.now())

    let pose = results.landmarks[0]
    console.log(results.landmarks[0])
}

/********************************************************************
 // START THE APP
 ********************************************************************/
if (navigator.mediaDevices?.getUserMedia) {
    createPoseLandmarker()
}

// Own added functions

async function addPose1(){
    results = await poseLandmarker.detectForVideo(video, performance.now())

    let pose = results.landmarks[0]

    let poseMapped = {}
    if(pose) {
        poseMapped.points = pose.flatMap(point => [point.x, point.y, point.z]);
        poseMapped.label = "pose1";
        allposes.push(poseMapped);
    }
}

async function addPose2(){
    results = await poseLandmarker.detectForVideo(video, performance.now())

    let pose = results.landmarks[0]

    let poseMapped = {}
    if(pose) {
        poseMapped.points = pose.flatMap(point => [point.x, point.y, point.z]);
        poseMapped.label = "pose2";
        allposes.push(poseMapped);
    }
}

async function addPose3(){
    results = await poseLandmarker.detectForVideo(video, performance.now())

    let pose = results.landmarks[0]

    let poseMapped = {}
    if(pose) {
        poseMapped.points = pose.flatMap(point => [point.x, point.y, point.z]);
        poseMapped.label = "pose3";
        allposes.push(poseMapped);
    }
}

async function addPose4(){
    results = await poseLandmarker.detectForVideo(video, performance.now())

    let pose = results.landmarks[0]

    let poseMapped = {}
    if(pose) {
        poseMapped.points = pose.flatMap(point => [point.x, point.y, point.z]);
        poseMapped.label = "pose4";
        allposes.push(poseMapped);
    }
}

async function train(){
    compressToJSON()
    allposes = allposes.toSorted(() => (Math.random() - 0.5))
    trainData = allposes.slice(0, Math.floor(allposes.length * 0.8));
    console.log(allposes)
    for (let pose of trainData) {
        nn.addData(pose.points, {label: pose.label});
    }

    await nn.normalizeData()
    await nn.train({ epochs: 50, learningRate: 0.2, hiddenUnits: 16 }, () => finishedTraining())
    async function finishedTraining(){
        await nn.save("model", () => console.log("model was saved!"))
        await testAccuracy(testData)

    }
}

async function testAccuracy() {
    let correctAnswers = 0

    allposes = await allposes.toSorted(() => (Math.random() - 0.5))
    testData = await allposes.slice(Math.floor(allposes.length * 0.8) + 1);
    console.log(testData)
    for await (let pose of testData) {
        const classifiedresults = await nn.classify(pose.points);

        if (classifiedresults[0].label === pose.label) {
            correctAnswers += 1
            console.log("correct")
        } else {
            console.log("incorrect")
        }
    }
    await console.log(correctAnswers);
    await console.log(testData.length);
    await console.log(correctAnswers / testData.length * 100);
}

async function compressToJSON() {
    const json = JSON.stringify(allposes.flatMap(pose => pose.points));
    const a = Object.assign(document.createElement("a"), {
        href: URL.createObjectURL(new Blob([json], { type: "application/json" })),
        download: "compressedPoses.json"
    });
    a.click();
}