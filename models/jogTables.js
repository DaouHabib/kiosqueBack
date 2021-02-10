const mongoose = require('mongoose');

const Schema = mongoose.Schema;

const jogSchema = new Schema({
    valuecm: {
        type: Number,
        required:true  
    },
    valueL: {
        type: Number,
        required:true  
    },
    capacity: {
        type: Number,
        required:true  
    }
   
});



const jogTables = mongoose.model('jogTables', jogSchema);
module.exports = jogTables; 