const mongoose = require('mongoose');

const Schema = mongoose.Schema;

const historiqueIndexSchema = new Schema({
    date: {
        type: Date,
        required:true  
    },
    typePoste: {
        type: String,
        required:true  
    },
    indexdebut: {
        type: Number,
        required:true  
    },
    indexfin: {
        type: Number,
        required:true  
    },
 
   
});



const historiqueIndex = mongoose.model('historiqueIndex', historiqueIndexSchema);
module.exports = historiqueIndex ; 