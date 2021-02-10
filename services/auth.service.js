const bcrypt = require('bcryptjs');
const { User } = require('../models/user');
const jwt = require('jsonwebtoken');
const { JWT_SECRET } = require('../config/config');

module.exports = {

    login: async (req, res, next) => {
        const foundUser = await User.findOne({ email: req.body.email });
        if (!foundUser) {
            res.status(400).json({ err: 'NOT FOUND EMAIL' });
        } else {
            const validpwd = await bcrypt.compare(req.body.password, foundUser.password);
            if (validpwd) {
                const token = jwt.sign({
                    _id:foundUser._id,firstName:foundUser.firstName,lastName: foundUser.lastName,Role:foundUser.Role,
                    iss: 'habibdaou',
                    sub: foundUser.id,
                    iat: new Date().getTime(),
                    exp: new Date().setDate(new Date().getDate() + 1)
                }, JWT_SECRET);
                res.status(200).json({ token :token });
            } else {
                res.status(400).json({ err: 'WRONG PWD' });
            }
        }
    }
}

