const mongoose = require('mongoose');

const Schema = mongoose.Schema;

const purchaseSchema = new Schema({
    date: Date,
    name: String,
    price: Number,
    paymentMethod: {
        type: String,
        enum : ['CARTE','ESPECE'],
        default: 'ESPECE'
    },
    user : {
        type: Schema.Types.ObjectId,
        ref: 'user'
    },
    category : {
        type: Schema.Types.ObjectId,
        ref: 'category'
    },
});

const Purchase = mongoose.model('purchase', purchaseSchema);
module.exports = Purchase;