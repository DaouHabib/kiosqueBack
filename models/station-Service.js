const mongoose = require('mongoose');

const Schema = mongoose.Schema;

const StationServiceSchema = new Schema({
    name: {
        type: String,
 
    },
    street: {
        type: String,
 
    },
    city: {
        type: String,
 
    },
    zipcode: {
        type: String,
 
    },
    department: {
        type: String,
 
    },
    post: {
        type: String,
 
    },
    tanks: [{
        type: Schema.Types.ObjectId,
        ref: "tank"
    }],
    pompes: [{
        type: Schema.Types.ObjectId,
        ref: "pompe"
    }],

});



const StationService = mongoose.model('stationService', StationServiceSchema);
module.exports = StationService; 