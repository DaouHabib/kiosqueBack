
var express = require('express');
var router = express.Router();
var fs = require('fs');
const tf = require('@tensorflow/tfjs')
var content = fs.readFileSync('data2019C.json');
var data2019 = JSON.parse(content);
var content1 = fs.readFileSync('data2019.json');
var predictData = JSON.parse(content1);
const vente = require('../models/vente');
var date = {day:Number,mounth:Number,year:Number};
var obj =[date];
const prepareData = async () => {
    const csv = await Papa.parsePromise(
      "https://raw.githubusercontent.com/curiousily/Linear-Regression-with-TensorFlow-js/master/src/data/housing.csv"
    );
  
    return csv.data;
  };
  const data = await prepareData();
  const createDataSets = (data, features, categoricalFeatures, testSize) => {
    const X = data.map(r =>
      features.flatMap(f => {
        if (categoricalFeatures.has(f)) {
          return oneHot(!r[f] ? 0 : r[f], VARIABLE_CATEGORY_COUNT[f]);
        }
        return !r[f] ? 0 : r[f];
      })
    );
  
    const X_t = normalize(tf.tensor2d(X));
  
    const y = tf.tensor(data.map(r => (!r.SalePrice ? 0 : r.SalePrice)));
  
    const splitIdx = parseInt((1 - testSize) * data.length, 10);
  
    const [xTrain, xTest] = tf.split(X_t, [splitIdx, data.length - splitIdx]);
    const [yTrain, yTest] = tf.split(y, [splitIdx, data.length - splitIdx]);
  
    return [xTrain, xTest, yTrain, yTest];
  };
  const VARIABLE_CATEGORY_COUNT = {
    OverallQual: 10,
    GarageCars: 5,
    FullBath: 4
  };
  const oneHot = (val, categoryCount) =>
  Array.from(tf.oneHot(val, categoryCount).dataSync());
  const normalize = tensor =>
  tf.div(
    tf.sub(tensor, tf.min(tensor)),
    tf.sub(tf.max(tensor), tf.min(tensor))
  );
  
const model = tf.sequential();

model.add(
  tf.layers.dense({
    inputShape: [xTrain.shape[1]],
    units: xTrain.shape[1]
  })
);

model.add(tf.layers.dense({ units: 1 }));
model.compile({
    optimizer: tf.train.sgd(0.001),
    loss: "meanSquaredError",
    metrics: [tf.metrics.meanAbsoluteError]
  });
  const trainLogs = [];
const lossContainer = document.getElementById("loss-cont");
const accContainer = document.getElementById("acc-cont");

await model.fit(xTrain, yTrain, {
  batchSize: 32,
  epochs: 100,
  shuffle: true,
  validationSplit: 0.1,
  callbacks: {
    onEpochEnd: async (epoch, logs) => {
      trainLogs.push({
        rmse: Math.sqrt(logs.loss),
        val_rmse: Math.sqrt(logs.val_loss),
        mae: logs.meanAbsoluteError,
        val_mae: logs.val_meanAbsoluteError
      });
      tfvis.show.history(lossContainer, trainLogs, ["rmse", "val_rmse"]);
      tfvis.show.history(accContainer, trainLogs, ["mae", "val_mae"]);
    }
  }
});
const trainLogs = [];
const lossContainer = document.getElementById("loss-cont");
const accContainer = document.getElementById("acc-cont");

await model.fit(xTrain, yTrain, {
  batchSize: 32,
  epochs: 100,
  shuffle: true,
  validationSplit: 0.1,
  callbacks: {
    onEpochEnd: async (epoch, logs) => {
      trainLogs.push({
        rmse: Math.sqrt(logs.loss),
        val_rmse: Math.sqrt(logs.val_loss),
        mae: logs.meanAbsoluteError,
        val_mae: logs.val_meanAbsoluteError
      });
      tfvis.show.history(lossContainer, trainLogs, ["rmse", "val_rmse"]);
      tfvis.show.history(accContainer, trainLogs, ["mae", "val_mae"]);
    }
  }
});

const [xTrainSimple, xTestSimple, yTrainSimple, yTestIgnored] = createDataSets(
    data,
    ["GrLivArea"],
    new Set(),
    0.1
  );
  
  const simpleLinearModel = await trainLinearModel(xTrainSimple, yTrainSimple);