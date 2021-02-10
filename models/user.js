const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const userSchema = new Schema({

    firstName: {
        type: String,
        required: true
    },
    lastName: {
        type: String,
        required: true
    },
    email: {
        type: String,
        required: true,
        unique: true,
        lowercase: true
    },
    password: {
        type: String,
    },
    phone: {
        type: Number,
        required: true
    },
    imageUrl: {
        type: String

    },
    civility: {
        type: String

    },
    post: {
        type: String
    },
    socialReason: {
        type: String
    },
    Role: {
        type: String,
        enum: [ "ADMIN", "USER","EMPLOYEE"]
    },
    stationServices: [{
        type: Schema.Types.ObjectId,
        ref: "stationService"
    }],
    creation_dt: { type: Date, require: true }

});
userSchema.methods.generateAuthToken = function(){

    const token =jwt.sign({_id:this._id,firstName:this.firstName,lastName: this.lastName,Role:this.Role},"jwtPrivateKey");
    return token;
}

const User = mongoose.model('user', userSchema);
module.exports.User = User;

