const  station  = require('../models/station-Service');
const  tank   = require('../models/tank');
const  {User}  = require('../models/user');
const   {getVentesyesterday }  = require('../services/pompe.service');
const   {newvente }  = require('../services/vente.service');
const vente = require('../models/vente');

const jogTables= require('../models/jogTables');
module.exports = {
        
     
        getAll: async (req, res, next) => {
                const stations = await station.find({});
                res.status(200).json(stations);
        },

        newstation: async (req, res, next) => {
                stations = new station(req.body);
                await stations.save();
                res.status(200).json(stations);
        },
     
        getStation: async (req, res, next) => {
                const stations = await station.findById(req.params.stationId);
                res.status(200).json(stations);
        },

        // PATCH || PUT
        updateStation: async (req, res, next) => {
               
                const newstation = req.body;
                const stations = await station.findByIdAndUpdate(req.params.stationId, newstation);
                res.status(200).json('success');
        },

        deleteStation: async (req, res, next) => {
                const stations = await station.findOneAndDelete(req.params.stationId).exec(function(err, item) {
                        if (err) {
                            return res.json({success: false, msg: 'Cannot remove item'});
                        }       
                        if (!item) {
                            return res.status(404).json({success: false, msg: 'Station not found'});
                        }  
                        res.json({success: true, msg: 'Station deleted.'});
                    });
               
        },
        deleteAll: async (req, res, next) => {
                const stations = await station.deleteMany();
                res.status(200).json('success');
        },

         getallStationByUser: async (req, res, next) => {
                const user = await User.findById(req.params.userId).populate("stationServices")
                res.status(200).json(user);
        },
        
        AddStationToUser: async (req, res, next) => {
                const user = await User.findById(req.body.id);
                const sts = new station(req.body.stationServices);
                sts.save();
                user.stationServices.push(sts);
                user.save();
                res.status(200).json(user);
        },

        addVentes: async (req, res, next) => {
                let result;
                let ventes; 
                const stations = await station.find({});
                stations.forEach(data=>{
                   result=getVentesyesterday(data._id).then(function(results) {
                           console.log(results);
                           ventes = new vente(results);
                           ventes.save();

                   });
                        
                })
                res.status(200).json(stations);

        }
      


}
