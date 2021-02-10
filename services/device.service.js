const device = require('../models/device');

module.exports = {


        getAll: async (req, res, next) => {
                const devices = await device.find({});
                res.status(200).json(devices);
        },

        newdevice: async (req, res, next) => {
                devices = new device(req.body);
                await devices.save();
                res.status(200).json(devices);
        },

        getdevice: async (req, res, next) => {
                const devices = await device.findById(req.params.deviceId);
                res.status(200).json(devices);
        },
        // PATCH || PUT
        updateDevice: async (req, res, next) => {
                const newdevice = req.body;
                const devices = await device.findByIdAndUpdate(req.params.deviceId, newdevice);
                res.status(200).json('success');
        },

        deletedevice: async (req, res, next) => {
                const devices = await device.findOneAndDelete(req.params.deviceId).exec(function(err, item) {
                        if (err) {
                            return res.json({success: false, msg: 'Cannot remove item'});
                        }       
                        if (!item) {
                            return res.status(404).json({success: false, msg: 'Device not found'});
                        }  
                        res.json({success: true, msg: 'Device deleted.'});
                });
        },
        deleteAll: async (req, res, next) => {
                const devices = await device.deleteMany();
                res.status(200).json('success');
        },

}
