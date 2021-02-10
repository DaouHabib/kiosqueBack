const mongoose = require('mongoose');
const User = require('../models/user');

const Schema = mongoose.Schema;

const categorySchema = new Schema({
    name: String,
    user : {
        type: Schema.Types.ObjectId,
        ref: 'user'
    },
});


categorySchema.pre('remove', function(next) {
    User.remove({categories: this._id}).exec();
    next();
});

const Category = mongoose.model('category', categorySchema);
module.exports = Category;