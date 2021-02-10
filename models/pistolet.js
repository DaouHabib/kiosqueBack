const mongoose = require('mongoose');

const Schema = mongoose.Schema;

const pistoletSchema = new Schema({
   
   
    name: {
        type: String,
       required:true
 
    },

    type: {
        type: String,
        enum: ["ESSENCE","DISEL"]
 
    },

    historiqueIndexes: [{
        type: Schema.Types.ObjectId,
        ref: 'historiqueIndex'
    }],

});

const pistolet = mongoose.model('pistolet', pistoletSchema);
module.exports = pistolet;