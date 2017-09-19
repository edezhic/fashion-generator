import { Array1D, Array2D, Array3D, Array4D, CheckpointLoader, Graph, NDArray, NDArrayInitializer, NDArrayMath, NDArrayMathGPU, Scalar, Session, Tensor } from 'deeplearn';

// Noise vector dimensions
const z_dim = 60;

// UI elements
const form = document.getElementById("generator");
const placeholder = document.getElementById("generator-loading-placeholder")
const control_panel = document.getElementById("generator-controls");
const controls = renderControls(control_panel, z_dim);
const results = document.getElementById("generator-results");

// Weights loader instance
const reader = new CheckpointLoader('../weights');
//const reader = new CheckpointLoader('http://cognitivechaos.com/wp-content/uploads/2017/fashion-gan/');

const math = new NDArrayMathGPU();

// Generator instance
var generator;

// Variables to measure generation speed
var t_start, t_end;

reader.getAllVariables().then(vars => {

  // Remove placeholder and show generator contents
  form.style.display = "block";
  placeholder.style.display = "none";

  // Build a Generator and generate first image
  generator = buildGenerator(math, vars);
  generateRandom();

  // Make buttons active
  document.getElementById("generate").addEventListener("click", generate);
  document.getElementById("generateRandom").addEventListener("click", generateRandom);
  document.getElementById("generateBatch").addEventListener("click", generateBatch);
  document.getElementById("clearResults").addEventListener("click", clearResults);

});

function buildGenerator(math: NDArrayMath, vars: { [varName: string]: NDArray }):
  (x: Array1D) => Array1D {
  // Get all weights
  // Fully-connected 0
  const fc0_bias = vars['generator/g_fc0/bias'] as Array1D;
  const fc0_w = vars['generator/g_fc0/Matrix'] as Array2D;
  // BatchNorm 0
  const bn0_beta = vars['generator/g_bn0/beta'] as Array1D;
  const bn0_gamma = vars['generator/g_bn0/gamma'] as Array1D;
  const bn0_mean = vars['generator/g_bn0/moving_mean'] as Array1D;
  const bn0_var = vars['generator/g_bn0/moving_variance'] as Array1D;
  // Fully-connected 1
  const fc1_bias = vars['generator/g_fc1/bias'] as Array1D;
  const fc1_w = vars['generator/g_fc1/Matrix'] as Array2D;
  // BatchNorm 1
  const bn1_beta = vars['generator/g_bn1/beta'] as Array1D;
  const bn1_gamma = vars['generator/g_bn1/gamma'] as Array1D;
  const bn1_mean = vars['generator/g_bn1/moving_mean'] as Array1D;
  const bn1_var = vars['generator/g_bn1/moving_variance'] as Array1D;
  // Fully-connected 2
  const fc2_bias = vars['generator/g_fc2/bias'] as Array1D;
  const fc2_w = vars['generator/g_fc2/Matrix'] as Array2D;
  // BatchNorm 2
  const bn2_beta = vars['generator/g_bn2/beta'] as Array1D;
  const bn2_gamma = vars['generator/g_bn2/gamma'] as Array1D;
  const bn2_mean = vars['generator/g_bn2/moving_mean'] as Array1D;
  const bn2_var = vars['generator/g_bn2/moving_variance'] as Array1D;
  // DeConv 1
  const dc1_bias = vars['generator/g_dc3/biases'] as Array1D;
  const dc1_w = vars['generator/g_dc3/w'] as Array4D;
  // BatchNorm 3
  const bn3_beta = vars['generator/g_bn3/beta'] as Array1D;
  const bn3_gamma = vars['generator/g_bn3/gamma'] as Array1D;
  const bn3_mean = vars['generator/g_bn3/moving_mean'] as Array1D;
  const bn3_var = vars['generator/g_bn3/moving_variance'] as Array1D;
  // DeConv 2
  const dc2_bias = vars['generator/g_dc4/biases'] as Array1D;
  const dc2_w = vars['generator/g_dc4/w'] as Array4D;

  const eps = Scalar.new(1e-6);

  return (x: Array1D): Array1D => {
    return math.scope(() => {
      // Generator network computations
      // fc0   = x*W + b
      const fully_connected0 = math.addStrict(math.vectorTimesMatrix(x, fc0_w), fc0_bias) as Array1D;
      // x_bar = (fc0 - mean) / sqrt(variance + epsilon)
      const x_bar0 = math.divideStrict(math.subStrict(fully_connected0, bn0_mean), math.sqrt(math.add(bn0_var, eps))) as Array1D;
      // bn0   = (x_bar ⋅ gamma) + beta
      const batch_norm0 = math.addStrict(math.multiplyStrict(x_bar0, bn0_gamma), bn0_beta) as Array1D;
      // h0    = ReLU(bn0)
      const hidden0 = math.relu(batch_norm0) as Array1D;
      // fc1   = h0*W + b
      const fully_connected1 = math.addStrict(math.vectorTimesMatrix(hidden0, fc1_w), fc1_bias) as Array1D;
      // x_bar = (fc1 - mean) / sqrt(variance + epsilon)
      const x_bar1 = math.divideStrict(math.subStrict(fully_connected1, bn1_mean), math.sqrt(math.add(bn1_var, eps))) as Array1D;
      // bn1   = (x_bar ⋅ gamma) + beta
      const batch_norm1 = math.addStrict(math.multiplyStrict(x_bar1, bn1_gamma), bn1_beta) as Array1D;
      // h1    = ReLU(bn0)
      const hidden1 = math.relu(batch_norm1) as Array1D;
      // fc2   = h0*W + b
      const fully_connected2 = math.addStrict(math.vectorTimesMatrix(hidden1, fc2_w), fc2_bias) as Array1D;
      // x_bar = (fc2 - mean) / sqrt(variance + epsilon)
      const x_bar2 = math.divideStrict(math.subStrict(fully_connected2, bn2_mean), math.sqrt(math.add(bn2_var, eps))) as Array1D;
      // bn2   = (x_bar ⋅ gamma) + beta
      const batch_norm2 = math.addStrict(math.multiplyStrict(x_bar2, bn2_gamma), bn2_beta) as Array1D;
      // h2    = ReLU(bn0)
      const hidden2 = math.relu(batch_norm2) as Array1D;
      // dc_in = h2.reshape (6272) => (7, 7, 128)
      const deconv1_input = hidden2.reshape([7, 7, 128]) as Array3D;
      // dc1   = transposed_convolution(dc_in, w) + b
      const deconv1 = math.add(math.conv2dTranspose(deconv1_input, dc1_w, [14, 14, 64], [2, 2], 'valid'), dc1_bias) as Array3D;
      // x_bar = (dc1 - mean) / sqrt(variance + epsilon)
      const x_bar3 = math.divide(math.sub(deconv1, bn3_mean), math.sqrt(math.add(bn3_var, eps))) as Array3D;
      // bn3   = (x_bar ⋅ gamma) + beta
      const batch_norm3 = math.add(math.multiply(x_bar3, bn3_gamma), bn3_beta) as Array3D;
      // h3    = ReLU(bn3)
      const hidden3 = math.relu(batch_norm3);
      // dc2   = transposed_convolution(h3, w) + b
      const deconv2 = math.add(math.conv2dTranspose(hidden3, dc2_w, [28, 28, 1], [2, 2], 'valid'), dc2_bias) as Array3D;
      // return  sigmoid(dc2)
      const result = math.sigmoid(deconv2);
      return result.reshape([784]) as Array1D;
    });
  };
}

function clearResults() {
  results.innerHTML = "";
}

function setControlsValues(z: Array1D) {
  const values = Array.prototype.slice.call(z.getValues());
  for (let i = 0; i < z_dim; i++) {
    controls[i].value = values[i];
  }
}

function getControlsValues() {
  const values = [];
  for (let i = 0; i < z_dim; i++) {
    values.push(controls[i].value);
  }
  return values;
}

function generate() {
  const z = Array1D.new(getControlsValues());
  math.scope(() => {
    t_start = performance.now();
    const canvas = renderImage(generator(z));
    t_end = performance.now();
    console.log("Image generation took " + (t_end - t_start) + " milliseconds.")
    results.appendChild(canvas);
  });
}

function generateRandom() {
  const z = getNoise();
  setControlsValues(z);
  generate();
}

function generateBatch() {
  for (let i = 0; i < 16; i++) {
    generateRandom();
  }
}

function getNoise() {
  return NDArray.randUniform<Array1D>([z_dim], -1, 1);
}

function renderControls(control_panel: HTMLElement, z_dim: number) {
  for (let i = 0; i < z_dim; i++) {
    control_panel.insertAdjacentHTML("beforeend",
      "<p class='generator-range-field'> <input type='range' id=" + i + " min='-1' max='1' step='0.00001' />  </p>"
    );
  }
  return control_panel.getElementsByTagName("input");
}

function renderImage(array: Array1D) {
  const width = 28;
  const height = 28;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const float32Array = array.getValues();
  const imageData = ctx.createImageData(width, height);
  for (let i = 0; i < float32Array.length; i++) {
    const j = i * 4;
    const value = Math.round(float32Array[i] * 255);
    imageData.data[j + 0] = value;
    imageData.data[j + 1] = value;
    imageData.data[j + 2] = value;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}
