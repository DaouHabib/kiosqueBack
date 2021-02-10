const historiqueTank = require('../models/historique-tank');
const tank = require('../models/tank');

module.exports = {


        getAll: async (req, res, next) => {
                const historiqueTanks = await historiqueTank.find({});
                res.status(200).json(historiqueTanks);
        },

        newhistoriqueTank: async (req, res, next) => {
                historiqueTanks = new historiqueTank(req.body);
                await historiqueTanks.save();
                res.status(200).json(historiqueTanks);
        },

        gethistoriqueTank: async (req, res, next) => {
                const historiqueTanks = await historiqueTank.findById(req.params.historiqueTankId);
                res.status(200).json(historiqueTanks);
        },
        // PATCH || PUT
        updatehistoriqueTank: async (req, res, next) => {
                const newhistoriqueTank = req.body;
                const historiqueTanks = await historiqueTank.findByIdAndUpdate(req.params.historiqueTankId, newhistoriqueTank);
                res.status(200).json('success');
        },

        deletehistoriqueTank: async (req, res, next) => {
                const historiqueTanks = await historiqueTank.findOneAndDelete(req.params.historiqueTankId).exec(function(err, item) {
                        if (err) {
                            return res.json({success: false, msg: 'Cannot remove item'});
                        }       
                        if (!item) {
                            return res.status(404).json({success: false, msg: 'Historique Tank not found'});
                        }  
                        res.json({success: true, msg: 'Historique tank deleted.'});
                });
        },
        deleteAll: async (req, res, next) => {
                const historiqueTanks = await historiqueTank.deleteMany();
                res.status(200).json('success');
        },
        AddHistoriqueToTank: async (req, res, next) => {
        const tanks = await tank.findById(req.params.idTank);
        const historiqueTanks = new historiqueTank(req.body);
        historiqueTanks.save();
        tanks.historiqueTank.push(historiqueTanks);
        tanks.save();
        res.status(200).json(historiqueTanks);
    },

    getHistoriqueByTank: async (req, res, next) => {
        const tanks = await tank.findById(req.params.historiqueTankId).populate("historiqueTank");
        res.status(200).json(tanks.historiqueTank);
    },

    getLastHistoriqueByTank: async (req, res, next) => {
         tanks = await tank.findById(req.params.historiqueTankId).populate("historiqueTank");
        var last ;
        if(tanks.historiqueTank.pop()!=undefined){

              last= tanks.historiqueTank.pop();  
        }
        else if(tanks.historiqueTank.pop()==undefined){last={value:0,date:null}}
        res.status(200).json(last);

    },
    getLastHistoriqueByStation: async (id) => {
        tanks = await tank.findById(id).populate("historiqueTank");
       var last ;
             last= tanks.historiqueTank.pop();  
       return last;

   },
    getLastHistoriqueByListTank: async (req, res, next) => {
        var tanks =[];
        tanks=req.body.tanks;
        var last =[];
        var tankis;
        tanks.forEach(async (element) => {
        last.push(element.historiqueTank.pop())  ;
        });
        res.status(200).json(last);
    },
        
}
