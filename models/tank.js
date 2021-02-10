const mongoose = require('mongoose');

const Schema = mongoose.Schema;

const TankSchema = new Schema({
    capacity: {
        type: Number,
        required: true
 
    },
    min: {
        type: Number,
        required: true
 
    },
    max: {
        type: Number,
        required: true
 
    },
    type: {
        type: String,
        enum: ["ESSENCE","DISEL"]
 
    },
    deviceRef:{
        type: Number,
        required: true
 
    },  
    jogTables: [{
        type: Schema.Types.ObjectId,
        ref: 'jogTables'
 
    }],

    historiqueTank: [{
        type: Schema.Types.ObjectId,
        ref: 'historiqueTank'
 
    }],
    ravitaillements: [{
        type: Schema.Types.ObjectId,
        ref: 'ravitaillement'
 
    }],

});



const Tank = mongoose.model('tank', TankSchema);
module.exports = Tank ; 