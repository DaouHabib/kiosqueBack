const mongoose = require('mongoose');
const { string } = require('joi');

const Schema = mongoose.Schema;

const AlertSchema = new Schema({
    title: {
        type: String,
        required:true  
        
    },
    message: {
        type: String,
        required:true  
    },
    date: {
        type: Date,
        required:true  
    },

    idUser: {
        type: String,
        required:true  
    },
});



const Alert = mongoose.model('alert', AlertSchema);
module.exports = Alert; 