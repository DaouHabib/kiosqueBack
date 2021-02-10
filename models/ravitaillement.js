const mongoose = require('mongoose');

const Schema = mongoose.Schema;

const RavitaillementSchema = new Schema({
  
    type: {
        type: String,
        enum: ["ESSENCE","DISEL"]
    },
    quantite:{
        type: Number,
        required: true
    },
    date: {
        type: Date,
        required: true
 
    },
});

const Ravitaillement = mongoose.model('ravitaillement', RavitaillementSchema);
module.exports = Ravitaillement;