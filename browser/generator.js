"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var deeplearn_1 = require("deeplearn");
var z_dim = 60;
var form = document.getElementById("generator");
var placeholder = document.getElementById("generator-loading-placeholder");
var control_panel = document.getElementById("generator-controls");
var controls = renderControls(control_panel, z_dim);
var results = document.getElementById("generator-results");
var reader = new deeplearn_1.CheckpointLoader('../weights');
var math = new deeplearn_1.NDArrayMathGPU();
var generator;
var t_start, t_end;
reader.getAllVariables().then(function (vars) {
    form.style.display = "block";
    placeholder.style.display = "none";
    generator = buildGenerator(math, vars);
    generateRandom();
    document.getElementById("generate").addEventListener("click", generate);
    document.getElementById("generateRandom").addEventListener("click", generateRandom);
    document.getElementById("generateBatch").addEventListener("click", generateBatch);
    document.getElementById("clearResults").addEventListener("click", clearResults);
});
function buildGenerator(math, vars) {
    var fc0_bias = vars['generator/g_fc0/bias'];
    var fc0_w = vars['generator/g_fc0/Matrix'];
    var bn0_beta = vars['generator/g_bn0/beta'];
    var bn0_gamma = vars['generator/g_bn0/gamma'];
    var bn0_mean = vars['generator/g_bn0/moving_mean'];
    var bn0_var = vars['generator/g_bn0/moving_variance'];
    var fc1_bias = vars['generator/g_fc1/bias'];
    var fc1_w = vars['generator/g_fc1/Matrix'];
    var bn1_beta = vars['generator/g_bn1/beta'];
    var bn1_gamma = vars['generator/g_bn1/gamma'];
    var bn1_mean = vars['generator/g_bn1/moving_mean'];
    var bn1_var = vars['generator/g_bn1/moving_variance'];
    var fc2_bias = vars['generator/g_fc2/bias'];
    var fc2_w = vars['generator/g_fc2/Matrix'];
    var bn2_beta = vars['generator/g_bn2/beta'];
    var bn2_gamma = vars['generator/g_bn2/gamma'];
    var bn2_mean = vars['generator/g_bn2/moving_mean'];
    var bn2_var = vars['generator/g_bn2/moving_variance'];
    var dc1_bias = vars['generator/g_dc3/biases'];
    var dc1_w = vars['generator/g_dc3/w'];
    var bn3_beta = vars['generator/g_bn3/beta'];
    var bn3_gamma = vars['generator/g_bn3/gamma'];
    var bn3_mean = vars['generator/g_bn3/moving_mean'];
    var bn3_var = vars['generator/g_bn3/moving_variance'];
    var dc2_bias = vars['generator/g_dc4/biases'];
    var dc2_w = vars['generator/g_dc4/w'];
    var eps = deeplearn_1.Scalar.new(1e-6);
    return function (x) {
        return math.scope(function () {
            var fully_connected0 = math.addStrict(math.vectorTimesMatrix(x, fc0_w), fc0_bias);
            var x_bar0 = math.divideStrict(math.subStrict(fully_connected0, bn0_mean), math.sqrt(math.add(bn0_var, eps)));
            var batch_norm0 = math.addStrict(math.multiplyStrict(x_bar0, bn0_gamma), bn0_beta);
            var hidden0 = math.relu(batch_norm0);
            var fully_connected1 = math.addStrict(math.vectorTimesMatrix(hidden0, fc1_w), fc1_bias);
            var x_bar1 = math.divideStrict(math.subStrict(fully_connected1, bn1_mean), math.sqrt(math.add(bn1_var, eps)));
            var batch_norm1 = math.addStrict(math.multiplyStrict(x_bar1, bn1_gamma), bn1_beta);
            var hidden1 = math.relu(batch_norm1);
            var fully_connected2 = math.addStrict(math.vectorTimesMatrix(hidden1, fc2_w), fc2_bias);
            var x_bar2 = math.divideStrict(math.subStrict(fully_connected2, bn2_mean), math.sqrt(math.add(bn2_var, eps)));
            var batch_norm2 = math.addStrict(math.multiplyStrict(x_bar2, bn2_gamma), bn2_beta);
            var hidden2 = math.relu(batch_norm2);
            var deconv1_input = hidden2.reshape([7, 7, 128]);
            var deconv1 = math.add(math.conv2dTranspose(deconv1_input, dc1_w, [14, 14, 64], [2, 2], 'valid'), dc1_bias);
            var x_bar3 = math.divide(math.sub(deconv1, bn3_mean), math.sqrt(math.add(bn3_var, eps)));
            var batch_norm3 = math.add(math.multiply(x_bar3, bn3_gamma), bn3_beta);
            var hidden3 = math.relu(batch_norm3);
            var deconv2 = math.add(math.conv2dTranspose(hidden3, dc2_w, [28, 28, 1], [2, 2], 'valid'), dc2_bias);
            var result = math.sigmoid(deconv2);
            return result.reshape([784]);
        });
    };
}
function clearResults() {
    results.innerHTML = "";
}
function setControlsValues(z) {
    var values = Array.prototype.slice.call(z.getValues());
    for (var i = 0; i < z_dim; i++) {
        controls[i].value = values[i];
    }
}
function getControlsValues() {
    var values = [];
    for (var i = 0; i < z_dim; i++) {
        values.push(controls[i].value);
    }
    return values;
}
function generate() {
    var z = deeplearn_1.Array1D.new(getControlsValues());
    math.scope(function () {
        t_start = performance.now();
        var canvas = renderImage(generator(z));
        t_end = performance.now();
        console.log("Image generation took " + (t_end - t_start) + " milliseconds.");
        results.appendChild(canvas);
    });
}
function generateRandom() {
    var z = getNoise();
    setControlsValues(z);
    generate();
}
function generateBatch() {
    for (var i = 0; i < 16; i++) {
        generateRandom();
    }
}
function getNoise() {
    return deeplearn_1.NDArray.randUniform([z_dim], -1, 1);
}
function renderControls(control_panel, z_dim) {
    for (var i = 0; i < z_dim; i++) {
        control_panel.insertAdjacentHTML("beforeend", "<p class='generator-range-field'> <input type='range' id=" + i + " min='-1' max='1' step='0.00001' />  </p>");
    }
    return control_panel.getElementsByTagName("input");
}
function renderImage(array) {
    var width = 28;
    var height = 28;
    var canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    var ctx = canvas.getContext('2d');
    var float32Array = array.getValues();
    var imageData = ctx.createImageData(width, height);
    for (var i = 0; i < float32Array.length; i++) {
        var j = i * 4;
        var value = Math.round(float32Array[i] * 255);
        imageData.data[j + 0] = value;
        imageData.data[j + 1] = value;
        imageData.data[j + 2] = value;
        imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
    return canvas;
}
