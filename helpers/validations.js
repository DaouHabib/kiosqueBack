const Joi = require('joi');

module.exports = {
    schemas: {
        authSchema: Joi.object().keys({
            email: Joi.string().email().required(),
            password: Joi.string().required()
        }),
        userSchema: Joi.object().keys({
            firstName: Joi.string().min(5).max(50).required(),
            lastName: Joi.string().min(5).max(50).required(),
            email: Joi.string().min(5).max(255).required().email(),
            password: Joi.string().min(5).max(255).required(),
            phone: Joi.number().required(),
            Role: Joi.string().min(5).max(255).required()
        }),
        idSchema: Joi.object().keys({
            param: Joi.string().regex(/^[0-9a-fA-F]{24}$/).required()
        })
    }
}