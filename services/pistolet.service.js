const pistolet = require('../models/pistolet');
const pompe = require('../models/pompe');

module.exports = {


    getAll: async (req, res, next) => {
        const pistolets = await pistolet.find({});
        res.status(200).json(pistolets);
    },

    newPistolet: async (req, res, next) => {
        pistolets = new pistolet(req.body);
        await pistolets.save();
        res.status(200).json(pistolets);
    },

    getPistolet: async (req, res, next) => {
        const pistolets = await pistolet.findById(req.params.pistoletId);
        res.status(200).json(pistolets);
    },

    // PATCH || PUT
    updatePistolet: async (req, res, next) => {
        const newpistolet = req.body;
        const pistolets = await pistolet.findByIdAndUpdate(req.params.pistoletId, newpistolet);
        res.status(200).json('success');
    },

    deletePistolet: async (req, res, next) => {
        const pistolets = await pistolet.findOneAndDelete(req.params.pistoletId).exec(function(err, item) {
            if (err) {
                return res.json({success: false, msg: 'Cannot remove item'});
            }       
            if (!item) {
                return res.status(404).json({success: false, msg: 'Pistolet not found'});
            }  
            res.json({success: true, msg: 'Pistolet deleted.'});
        });
    },
    deleteAll: async (req, res, next) => {
        const pistolets = await pistolet.deleteMany();
        res.status(200).json('success');
    },
    AddPistoletByPompe: async (req, res, next) => {
        const pompes = await pompe.findById(req.body.id);
        const pistoletes = new pistolet(req.body.Pistolet);
        pistoletes.save();
        pompes.pistolets.push(pistoletes);
        pompes.save();
        res.status(200).json(pistoletes);
    },

    getPistoletsByPompe: async (req, res, next) => {
        const pompes = await pompe.findById(req.params.pompeId).populate("pistolets");
        res.status(200).json(pompes);
    },
}
