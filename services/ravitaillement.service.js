const ravitaillement = require('../models/ravitaillement');
const tankService = require('../services/tank.service');
const Tank = require('../models/tank');

module.exports = {


    getAll: async (req, res, next) => {
        const ravitaillements = await ravitaillement.find({});
        res.status(200).json(ravitaillements);
    },

    newravitaillement: async (req, res, next) => {
        ravitaillements = new ravitaillement(req.body);
        await ravitaillements.save();
        res.status(200).json(ravitaillements);
    },

    getravitaillement: async (req, res, next) => {
        const ravitaillements = await ravitaillement.findById(req.params.ravitaillementId);
        res.status(200).json(ravitaillements);
    },

    // PATCH || PUT
    updateravitaillement: async (req, res, next) => {
        const newravitaillement = req.body;
        const ravitaillements = await ravitaillement.findByIdAndUpdate(req.params.ravitaillementId, newravitaillement);
        res.status(200).json('success');
    },

    deleteravitaillement: async (req, res, next) => {
        const ravitaillements = await ravitaillement.findOneAndDelete(req.params.ravitaillementId).exec(function(err, item) {
            if (err) {
                return res.json({success: false, msg: 'Cannot remove item'});
            }       
            if (!item) {
                return res.status(404).json({success: false, msg: 'Ravitaillement not found'});
            }  
            res.json({success: true, msg: 'Ravitaillement deleted.'});
        });
    },
    
    deleteAll: async (req, res, next) => {
        const ravitaillements = await ravitaillement.deleteMany();
        res.status(200).json('success');
    },
    getBytank: async (req, res, next) => {
      let tank ;
      tank = await Tank.findById(req.params.id).populate("ravitaillements");
      res.status(200).json(tank.ravitaillements);
    },
    addBytank: async (req, res, next) => {
        tank = await Tank.findById(req.params.id).populate("ravitaillements");
        ravitaillements = new ravitaillement(req.body);
         await ravitaillements.save();
          tank.ravitaillements.push(ravitaillements);
         tank.save();
         res.status(200).json(tank);
      },
}
