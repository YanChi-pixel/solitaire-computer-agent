/* Card recognition demo: ONNX Runtime Web, everything client-side.
 *
 * The same MobileNet v2 that the agent uses on screen: model.onnx was exported
 * from model.pth (see hf/export_onnx.py) with softmax already inside the graph,
 * so the JS only has to build the input tensor.
 */

const SIZE = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const SUITS = { hearts: "♥", diamonds: "♦", clubs: "♣", spades: "♠" };
const ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const previewEl = document.getElementById("preview");
const dropEl = document.getElementById("drop");
const fileEl = document.getElementById("file");
const examplesEl = document.getElementById("examples");

let session = null;
let classes = null;

function status(text, busy) {
  statusEl.textContent = text;
  statusEl.className = busy ? "busy" : "";
}

function pretty(className) {
  const [rank, suit] = className.split("_");
  return rank + (SUITS[suit] || suit);
}

async function boot() {
  ort.env.wasm.wasmPaths = ORT_CDN;
  ort.env.wasm.numThreads = 1; // no cross-origin isolation on static hosting
  try {
    classes = (await (await fetch("classes.json")).json()).classes;
    session = await ort.InferenceSession.create("model.onnx", {
      executionProviders: ["wasm"],
    });
    status("Model ready — drop a card image.", false);
  } catch (error) {
    status("Could not load the model: " + error.message, false);
  }
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("could not read that image"));
    img.src = src;
  });
}

/* Resize to 224×224 on a canvas, then CHW float32 with ImageNet normalisation —
 * exactly what torchvision does before the model in the Python pipeline. */
function toTensor(img) {
  const canvas = document.createElement("canvas");
  canvas.width = SIZE;
  canvas.height = SIZE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, SIZE, SIZE);
  const { data } = ctx.getImageData(0, 0, SIZE, SIZE);

  const plane = SIZE * SIZE;
  const input = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    input[i] = (data[i * 4] / 255 - MEAN[0]) / STD[0];
    input[plane + i] = (data[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
    input[2 * plane + i] = (data[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
  }
  return new ort.Tensor("float32", input, [1, 3, SIZE, SIZE]);
}

function render(probs) {
  const order = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]);
  let html = "";
  for (const index of order.slice(0, 5)) {
    const percent = (probs[index] * 100).toFixed(1);
    html +=
      '<div class="bar"><div class="row"><span class="name">' +
      pretty(classes[index]) +
      '</span><span class="pct">' +
      percent +
      '%</span></div><div class="track"><div class="fill" style="width:' +
      percent +
      '%"></div></div></div>';
  }
  resultsEl.innerHTML = html;
}

async function predict(src) {
  if (!session || !classes) {
    status("Model is still loading…", true);
    return;
  }
  try {
    status("Recognising…", true);
    const img = await loadImage(src);
    previewEl.innerHTML = "";
    const shown = img.cloneNode();
    previewEl.appendChild(shown);

    const output = await session.run({ input: toTensor(img) });
    render(output[session.outputNames[0]].data);
    status("Done.", false);
  } catch (error) {
    status("Error: " + error.message, false);
  }
}

dropEl.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropEl.classList.add("over");
});
dropEl.addEventListener("dragleave", () => dropEl.classList.remove("over"));
dropEl.addEventListener("drop", (event) => {
  event.preventDefault();
  dropEl.classList.remove("over");
  const file = event.dataTransfer.files[0];
  if (file) predict(URL.createObjectURL(file));
});
fileEl.addEventListener("change", () => {
  if (fileEl.files[0]) predict(URL.createObjectURL(fileEl.files[0]));
});

(async () => {
  await boot();
  for (const name of [
    "king_of_spades.png",
    "ace_of_hearts.png",
    "eight_of_diamonds.png",
    "queen_of_clubs.png",
  ]) {
    const url = "examples/" + name;
    try {
      await loadImage(url);
    } catch (error) {
      continue;
    }
    const button = document.createElement("button");
    button.title = name.replace(".png", "").replace(/_/g, " ");
    const thumb = document.createElement("img");
    thumb.src = url;
    thumb.alt = button.title;
    button.appendChild(thumb);
    button.addEventListener("click", () => predict(url));
    examplesEl.appendChild(button);
  }
})();
