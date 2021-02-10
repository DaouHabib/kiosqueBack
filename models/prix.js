const mongoose = require('mongoose');

const Schema = mongoose.Schema;

const prixSchema = new Schema({
    prixLittreSp: {
        type: Number,
        required: true
 
    },
    prixLittreD: {
        type: Number,
        required: true
 
    },
    prixLittreDss: {
        type: Number,
        required: true
 
    },
    date: {
        type: Date,
        required: true
 
    },
});

const prix = mongoose.model('prix', prixSchema);
module.exports = prix;