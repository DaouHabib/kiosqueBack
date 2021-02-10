const mongoose = require('mongoose');

const Schema = mongoose.Schema;

const venteSchema = new Schema({
    totalEssence: {
        type: [],
    },
    totalDisel: {
        type:[],
    },
    date: {
        type: String,
 
    },
    stationId:{
        type: String,

    },
    totalpost1:{
        type: Number,

    },  totalpost2:{
        type: Number,

    },  totalpost3:{
        type: Number,

    },
});

const vente = mongoose.model('vente', venteSchema);
module.exports = vente;