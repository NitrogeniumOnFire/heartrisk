
// Lightweight in-browser NN forward pass using embedded model.json
const modelMeta = JSON.parse(document.getElementById('model-json').textContent);
const featureNames = modelMeta.feature_names;
const means = modelMeta.means;
const stds = modelMeta.stds;
const weights = modelMeta.weights;
const activations = modelMeta.layers.map(l => l.activation || 'linear');

function relu(x){ return x>0?x:0; }
function sigmoid(x){ return 1/(1+Math.exp(-x)); }

function matVecMul(W, x){ // W is 2D: rows = input_dim, cols = output_dim in our export; we need to compute x dot W + b
  // W: [in_dim][out_dim]; x: [in_dim]
  const out_dim = W[0].length;
  const out = new Array(out_dim).fill(0.0);
  for(let i=0;i<x.length;i++){
    const xi = x[i];
    const row = W[i];
    for(let j=0;j<out_dim;j++){
      out[j] += row[j]*xi;
    }
  }
  return out;
}

function addBias(vec, b){
  return vec.map((v,i)=>v + b[i]);
}

function applyActivation(vec, name){
  if(name === 'relu') return vec.map(v=>relu(v));
  if(name === 'sigmoid') return vec.map(v=>sigmoid(v));
  return vec;
}

function predict(prototypeInputs){
  // prototypeInputs: object feature->raw value (numbers)
  // build normalized input vector following means/stds and feature order
  const x = featureNames.map((f,i)=>{
    const v = parseFloat(prototypeInputs[f]);
    return isNaN(v)?0.0: (v - means[i]) / stds[i];
  });
  // Forward through layers
  let out = x;
  for(let li=0; li<weights.length; li++){
    const W = weights[li].W;
    const b = weights[li].b;
    out = matVecMul(W, out);
    out = addBias(out, b);
    out = applyActivation(out, activations[li]);
  }
  // final out is array of length 1 (sigmoid), probability in [0,1]
  let prob = out[0];
  if(typeof prob !== 'number') prob = parseFloat(prob);
  prob = Math.min(1, Math.max(0, prob));
  return prob;
}

// Build UI
const inputsGrid = document.getElementById('inputs-grid');
featureNames.forEach(fname => {
  const div = document.createElement('div');
  div.className = 'input';
  const label = document.createElement('label');
  label.textContent = fname.replace(/_/g,' ');
  const inp = document.createElement('input');
  inp.type = 'number';
  inp.id = 'f__' + fname;
  inp.placeholder = 'number';
  div.appendChild(label);
  div.appendChild(inp);
  inputsGrid.appendChild(div);
});

document.getElementById('predict-btn').addEventListener('click', ()=>{
  const vals = {};
  featureNames.forEach(f=>{
    vals[f] = document.getElementById('f__'+f).value;
  });
  const p = predict(vals);
  const pct = Math.round(p*100);
  const resDiv = document.getElementById('result');
  resDiv.classList.remove('hidden');
  resDiv.innerHTML = `<strong>Estimated risk:</strong> ${pct}% <br/><small>Probability: ${p.toFixed(3)}</small>`;
});

document.getElementById('fill-sample').addEventListener('click', ()=>{
  // fill with dataset means as sample
  featureNames.forEach((f,i)=>{
    document.getElementById('f__'+f).value = means[i].toFixed(2);
  });
});
