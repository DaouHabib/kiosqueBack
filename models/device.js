const mongoose = require('mongoose');

const Schema = mongoose.Schema;

const DeviceSchema = new Schema({
    ref: {
        type: Number,
        required:true  ,
        unique:true
    },
    value: {
        type: Number,
        required:true  
    },
    date: {
        type: Date,
        required:true  
    },
   
});



const Device = mongoose.model('device', DeviceSchema);
module.exports = Device; 