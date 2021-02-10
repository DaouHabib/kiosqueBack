const historiqueIndexe = require('../models/historique_indexes');
const pistolet = require('../models/pistolet');

module.exports = {


        getAll: async (req, res, next) => {
                const historiqueIndexes = await historiqueIndexe.find({});
                res.status(200).json(historiqueIndexes);
        },

        newhistoriqueIndexe: async (req, res, next) => {
                historiqueIndexes = new historiqueIndexe(req.body);
                await historiqueIndexes.save();
                res.status(200).json(historiqueIndexes);
        },

        gethistoriqueIndexe: async (req, res, next) => {
                const historiqueIndexes = await historiqueIndexe.findById(req.params.historiqueIndexeId);
                res.status(200).json(historiqueIndexes);
        },
        // PATCH || PUT
        updatehistoriqueIndexe: async (req, res, next) => {
                const newhistoriqueIndexe = req.body;
                const historiqueIndexes = await historiqueIndexe.findByIdAndUpdate(req.params.historiqueIndexeId, newhistoriqueIndexe);
                res.status(200).json('success');
        },

        deletehistoriqueIndexe: async (req, res, next) => {
                const historiqueIndexes = await historiqueIndexe.findOneAndDelete(req.params.historiqueIndexeId).exec(function(err, item) {
                        if (err) {
                            return res.json({success: false, msg: 'Cannot remove item'});
                        }       
                        if (!item) {
                            return res.status(404).json({success: false, msg: 'Historique not found'});
                        }  
                        res.json({success: true, msg: 'Historique deleted.'});
                });
        },
        deleteAll: async (req, res, next) => {
                const historiqueIndexes = await historiqueIndexe.deleteMany();
                res.status(200).json('success');
        },
        AddHistoriqueToTank: async (req, res, next) => {
        const pistolets = await pistolet.findById(req.params.idpistolet);
        const historiqueIndexes = new historiqueIndexe(req.body);
        historiqueIndexes.save();
        pistolets.historiqueIndexes.push(historiqueIndexes);
        pistolets.save();
        res.status(200).json(historiqueIndexes);
    },

    getHistoriqueByTank: async (req, res, next) => {
        const pistolets = await pistolet.findById(req.params.historiqueIndexeId).populate("historiqueIndexes");
        res.status(200).json(pistolets);
    },
    getVentesOfyesterday: async (req, res, next) => {
        let result = [];
        const pistolets = await pistolet.findById(req.params.idpistolet).populate("historiqueIndexes");
        if(pistolets.historiqueIndexes.length !=0 && pistolet.historiqueIndexes){
                console.log(pistolets.historiqueIndexes.length);
                pistolets.historiqueIndexes= pistolets.historiqueIndexes.slice(0).slice(-3)
}

pistolets.historiqueIndexes.forEach(element => {
        const date =new Date(element.date).toISOString().slice(0,10);
result.push({
        id: element._id,
        typePoste : element.typePoste,
        date:element.date,
        Total :element.indexfin-element.indexdebut,
})
       console.log(date); 
});

        res.status(200).json(result);
    },

        
}
