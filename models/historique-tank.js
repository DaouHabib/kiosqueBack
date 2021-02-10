const mongoose = require('mongoose');

const Schema = mongoose.Schema;

const historiqueTankSchema = new Schema({
   
    value: {
        type: Number,
        required:true  
    },
    date: {
        type: Date,
        required:true  
    },
 
   
});



const historiqueTank = mongoose.model('historiqueTank', historiqueTankSchema);
module.exports = historiqueTank; 