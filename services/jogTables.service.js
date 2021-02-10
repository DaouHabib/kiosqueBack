const jogTables = require('../models/jogTables');
module.exports = {

    getAll: async (req, res, next) => {
        const jogTable = await jogTables.find({});
        res.status(200).json(jogTable);
    },
    newjogTable: async (req, res, next) => {

    },
    getjogTable: async (req, res, next) => {
        const jogTable = await jogTables.findById(req.params.jogId);
        res.status(200).json(jogTable);
    },

    deletedevice: async (req, res, next) => {
        const jogTable = await jogTables.findOneAndDelete(req.params.jogId).exec(function(err, item) {
            if (err) {
                return res.json({success: false, msg: 'Cannot remove item'});
            }       
            if (!item) {
                return res.status(404).json({success: false, msg: 'Jog not found'});
            }  
            res.json({success: true, msg: 'Jog deleted.'});
        });
     
    },
    deleteAll: async (req, res, next) => {
        const jogTable = await jogTables.deleteMany();
        res.status(200).json('success');
    },

}
