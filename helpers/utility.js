const Joi = require('joi');
const bcrypt = require('bcryptjs');

module.exports = {
    hashedPwd : async (password) => {
        try {
            const salt = await bcrypt.genSalt(10);
            return await bcrypt.hash(password, salt);
        } catch (error) {
            throw  new Error('hashing failed');
    
        }
    },
    validateParam: (schema, name) => {
        return (req, res, next) => {
            const result = Joi.validate({ param: req['params'][name] }, schema)
            if (result.error) {
                return res.status(400).json(result.error);
            } else {
                if (!req.value) {
                    req.value = {};
                    if (!req.value['params'])
                        req.value['params'] = {}


                    req.value['params'][name] = result.value.param;
                    next();
                }
            }
        }
    },


    validateBody: (schema) => {
        return (req, res, next) => {
            const result = Joi.validate(req.body, schema)
            if (result.error) {
                return res.status(400).json(result.error);
            } else {
                if (!req.value) {
                    req.value = {};
                    if (!req.value['body'])
                        req.value['body'] = {}
                    req.value['body'] = result.value;
                    next();
                }
            }
        }
    },
    
 pagination:(model)=> {
    return async (req, res, next) => {
        const page = parseInt(req.params.page);
        const limit = parseInt(req.params.limit);
        const startIndex = (page - 1) * limit;
        const endIndex = page * limit;
        console.log(model)
        const results = {}
        if (endIndex < await model.countDocuments().exec()) {

            results.next = {
                page: page + 1,
                limit: limit
            }

        }


        if (startIndex > 0) {
            results.previous = {
                page: page - 1,
                limit: limit
            }
        }
        try {
            results.result = await model.find().limit(limit).skip(startIndex).exec()
            res.pagination = results;

            next()
        }

        catch (e) {
            res.status(500).json({ message: e.message })

        }
console.log(res.pagination);
    }
}
}