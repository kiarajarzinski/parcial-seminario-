// ============================================================
//  EJERCICIO B1  -  Formula objetivo:  y = 2x + 8
//  Backend Node.js  |  Puerto 8088


const http = require('http');
const tf   = require('@tensorflow/tfjs-node');


let trainedModel = null;


const XS_DATA = [-1,  0,  1, 4];
const YS_DATA = [-3, -1,  1, 5,  7];   

async function trainModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
  const xs = tf.tensor2d(XS_DATA, [XS_DATA.length, 1]);
  const ys = tf.tensor2d(YS_DATA, [YS_DATA.length, 1]);
  await model.fit(xs, ys, { epochs: 250 });
  xs.dispose();
  ys.dispose();
  return model;
}

const server = http.createServer(async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin',  '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Content-Type', 'application/json');

  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

  const url = req.url.split('?')[0];

  if (req.method === 'POST' && url === '/train') {
    try {
      console.log('Entrenando modelo...');
      trainedModel = await trainModel();
      console.log('Modelo entrenado.');
      res.writeHead(200);
      res.end(JSON.stringify({ status: 'ok', message: 'Modelo entrenado' }));
    } catch (err) {
      res.writeHead(500);
      res.end(JSON.stringify({ status: 'error', message: err.message }));
    }
    return;
  }

  if (req.method === 'POST' && url === '/predict') {
    if (!trainedModel) {
      res.writeHead(400);
      res.end(JSON.stringify({ status: 'error', message: 'El modelo no fue entrenado aun' }));
      return;
    }
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      try {
        const { x } = JSON.parse(body);
        const input  = tf.tensor2d([x], [1, 1]);
        const output = trainedModel.predict(input);
        const result = output.dataSync()[0];
        input.dispose();
        output.dispose();
        res.writeHead(200);
        res.end(JSON.stringify({ status: 'ok', x, y: result }));
      } catch (err) {
        res.writeHead(400);
        res.end(JSON.stringify({ status: 'error', message: err.message }));
      }
    });
    return;
  }

  res.writeHead(404);
  res.end(JSON.stringify({ status: 'error', message: 'Ruta no encontrada' }));
});

server.listen(......, () => {
  console.log('Servidor B1  (y = 2x + 8)  en http://localhost:' + PORT);
  console.log('Rutas: POST /train   POST /predict');
});