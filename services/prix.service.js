const prix = require('../models/prix');

module.exports = {


    getAll: async (req, res, next) => {
        const prixT = await prix.find({});
        res.status(200).json(prixT);
    },

    newprix: async (req, res, next) => {
        prixT = new prix(req.body);
        await prixT.save();
        res.status(200).json(prixT);
    },

    getprix: async (req, res, next) => {
        const prixT = await prix.findById(req.params.prixId);
        res.status(200).json(prixT);
    },

    // PATCH || PUT
    updateprix: async (req, res, next) => {
        const newprix = req.body;
        const prixT = await prix.findByIdAndUpdate(req.params.prixId, newprix);
        res.status(200).json('success');
    },

    deleteprix: async (req, res, next) => {
        const prixT = await prix.findOneAndDelete(req.params.prixId).exec(function(err, item) {
            if (err) {
                return res.json({success: false, msg: 'Cannot remove item'});
            }       
            if (!item) {
                return res.status(404).json({success: false, msg: 'Prix not found'});
            }  
            res.json({success: true, msg: 'Prix deleted.'});
        });
        res.status(200).json('success');
    },
    
    deleteAll: async (req, res, next) => {
        const prixT = await prix.deleteMany();
        res.status(200).json('success');
    },

}
