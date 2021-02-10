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


router.get('/VenteAlert',(req,res)=>{
    vente.find(function(err, data){
    if(err){            
        console.log(err);
    }
    obj= JSON.stringify(data);
res.sendStatus(201);
})   

});
const trainData = tf.tensor2d(predictData.map(item => [
  new Date( item.date),
]))

const outputData = tf.tensor2d(predictData.map(item => [
  item.Vente
]))

const testingData = tf.tensor2d(data2019.map(item => [
  new Date( item.date),
]))

const model = tf.sequential()

model.add(tf.layers.dense({
  inputShape: [1],
  activation: "linear",
  units: 6
}))

model.add(tf.layers.dense({
  inputShape: [365],
  activation: "linear",
  units: 1
})) 


model.add(tf.layers.dense({
  activation: "linear",
  units: 1
}))

model.compile({
  loss: "meanSquaredError",
  optimizer: tf.train.adam(.06),
})


model.fit(trainData, outputData, {epochs: 100,shuffle:true}).then((history) => console.log(history))

const data = tf.tensor([1,2,3,4]);

router.get('/predict',function (req, res, next){

data.print();
  res.send(model.predict(testingData).dataSync())
})

module.exports = router;

