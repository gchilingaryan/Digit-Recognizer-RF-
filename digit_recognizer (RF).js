let table;
let generation = 1;
let lr = 0.0001;
let gamma = 0.9;
let epsilon = 0.95;
let epsilon_min = 0.01;
let epsilon_decay = 0.995;
let memory = [];
let pixels = [];
let labels = [];
let action = 0;
let reward = 0;
let done = false;
let model = network();
let target_model = network();
let score_tag = document.getElementById("score");
let generation_tag = document.getElementById("generation");

function preload() {
  table = loadTable('train.csv', 'csv', 'header');
}

function setup() {
  let temp_pixels = [];
  for (let i = 0; i < table.rows.length; i++) {
    labels.push(Number(table.rows[i].obj.label));
  }
  for (let i = 0; i < table.rows.length; i++) {
    for (j in table.rows[i].obj){
      if (j != "label"){
      temp_pixels.push((Number(table.rows[i].obj[j])) / 255.0);
      }
    }
    pixels.push(temp_pixels);
  temp_pixels = [];
  }
  main();
}

function show(x) {
  let all_data = [];
  let img = get();
  img.resize(28, 28);
  img.loadPixels();
  for (let i = 0; i < 784; i++) {
    img.pixels[i * 4] = pixels[x][i];
    let r = img.pixels[i * 4];
    let g = img.pixels[i * 4 + 1];
    let b = img.pixels[i * 4 + 2];
    let a = img.pixels[i * 4 + 3];
    all_data.push(r + g + b + a);
  }
  
  let createGroupedArray = function(arr, chunkSize) {
    let groups = [];
    for (let i = 0; i < arr.length; i += chunkSize) {
      groups.push(arr.slice(i, i + chunkSize));
    }
    return groups;
}

let zdata = createGroupedArray(all_data, 28);

let data = [
  {
    z: zdata,
    type: 'heatmap',
    colorscale: "Greys"
  }
];
Plotly.newPlot('myDiv', data);
}

function remember(cur_state, action, reward, new_state, done) {
  memory.push([cur_state, action, reward, new_state, done]);
}

function network() {
  let model = tf.sequential();

  model.add(tf.layers.conv2d({inputShape: [28, 28, 1], filters: 64, kernelSize: (3, 3), activation: "relu"}));
  model.add(tf.layers.maxPooling2d({poolSize: [2 ,2]}));
  model.add(tf.layers.conv2d({filters: 128, kernelSize: (3, 3), activation: "relu"}));
  model.add(tf.layers.maxPooling2d({poolSize: [2 ,2]}));
  model.add(tf.layers.conv2d({filters: 64, kernelSize: (3, 3), activation: "relu"}));
  model.add(tf.layers.maxPooling2d({poolSize: [2 ,2]}));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({units: 512, activation: "relu"}));
  model.add(tf.layers.dense({units: 10, activation: "softmax"}));

  model.compile({loss: "meanSquaredError", optimizer: tf.train.adam(lr)});

  return model;
}

async function train() {
  let batch_size = 32;

  if (memory.length < batch_size){
    return
  }

  samples = _.sample(memory, batch_size);
  for (sample of samples) {
    let cur_state = sample[0];
    let action = sample[1];
    let reward = sample[2]; 
    let new_state = sample[3];
    let done = sample[4];
    target = target_model.predict(tf.tensor4d(cur_state, [1, 28, 28,1]));
    if (done) {
      target.flatten().dataSync()[action] = reward;
    }
    else {
    Q_future = math.max(Array.from((target_model.predict(tf.tensor4d(new_state, [1, 28, 28,1])).flatten()).dataSync()));
    target.flatten().dataSync()[action] = reward + Q_future * gamma;
    }
  await model.fit(tf.tensor4d(cur_state, [1, 28, 28,1]), target, epochs=1, verbose=0);
  }
}

function target_train(){
  weights = model.getWeights();
  target_weights = target_model.getWeights();
  for (let i = 0; i < target_weights.length; i++){
    target_weights[i] = weights[i];
  }
  target_model.setWeights(target_weights);
}

function act(cur_state){
  epsilon *= epsilon_decay;
  epsilon = math.max(epsilon_min, epsilon);
  if (math.random() <= epsilon) {
    return math.round(math.random(0, 9));
  }
  console.log('model guessed');
  return ((model.predict(tf.tensor4d(cur_state, [1, 28, 28,1]))).flatten().argMax().dataSync())[0];
}

async function main() {
  console.log("first guess");
  while (true) {
    let score = 0;
    cur_state = pixels[0];
    for (let i = 0; i < 50; i++) {
      await show(i);
      action = act(cur_state);
      console.log("guessed:", action);
      if (i === 49) {
        done = true;
      }
      if (labels[i] === action) {
        reward = 1;
        score += 1;
        score_tag.innerHTML = "Score: " + score;
      }
      else {
        reward = -1;
      }
      console.log("reward:", reward);
      console.log("next guess");
      try {
      new_state = pixels[i + 1];
      }
      catch {
        continue;
      }
      remember(cur_state, action, reward, new_state, done);
      await train();
      await target_train();
      cur_state = new_state;
      console.log(i);
      if (done) {
        break;
      }
    }
    console.log("score:", score);
    done = false;
    generation += 1;
    score_tag.innerHTML = "Score: " + 0;
    generation_tag.innerHTML = "Generation: " + generation;
    if (score >= 47){
      break;
    }
  }
}