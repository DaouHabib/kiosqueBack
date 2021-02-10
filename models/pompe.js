const mongoose = require('mongoose');

const Schema = mongoose.Schema;

const pompeSchema = new Schema({
    name: {
        type: String,
        required: true
    },
  
    pistolets: [{
        type: Schema.Types.ObjectId,
        ref: "pistolet"
    }],
});

const pompe = mongoose.model('pompe', pompeSchema);
module.exports = pompe;