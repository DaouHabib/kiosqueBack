const tank = require('../models/tank');
const device = require('../models/device');
const station = require('../models/station-Service');
const jogTables = require('../models/jogTables');
const historique = require('../models/historique-tank');
const webpush = require('web-push');
const privateKey= 'XAtPjgxcdQgfPc7-4uVG3LbajICCXQmHW3QpffHa3sc';
const publicKey= 'BAkmXAVNU7muOJ_ZUpD3ca5rzKr-CoaPusTzGQD96VuEeoYtCVrOXg8Ov3Wg0fVNm3Xo7DbIituY10sb7uh5LUg';
webpush.setVapidDetails("mailTo:test@test.com",publicKey,privateKey)
module.exports = {


        getAll: async (req, res, next) => {
                const tanks = await tank.find({});
                res.status(200).json(tanks);
        },

        newTank: async (req, res, next) => {
                tanks = new tank(req.body);
                await tanks.save();
                res.status(200).json(tanks);
        },

        getTank: async (req, res, next) => {


                const tanks = await tank.findById(req.params.tankId);
                res.status(200).json(tanks);
        },

        // PATCH || PUT
        updateTank: async (req, res, next) => {

                const newtank = req.body;
                const tanks = await tank.findByIdAndUpdate(req.params.tankId, newtank);
                res.status(200).json('success');
        },

        deleteTank: async (req, res, next) => {
                const tanks = await tank.findById(req.params.tankId);
                tanks.jogTables.forEach(res=>{  
                        jogTables.findByIdAndDelete(res._id);   

                })
                const tanksss = await tank.findOneAndDelete(req.params.tankId).exec(function(err, item) {
                        if (err) {
                            return res.json({success: false, msg: 'Cannot remove item'});
                        }       
                        if (!item) {
                            return res.status(404).json({success: false, msg: 'Tank not found'});
                        }  
                        res.json({success: true, msg: 'Tank deleted.'});
                    });
        },
        deleteAll: async (req, res, next) => {
                const tanks = await tank.deleteMany();
                res.status(200).json('success');
        },
        AddDevice: async (req, res, next) => {
                const tanks = await tank.findById(req.body.id);
                const devices = new device(req.body.device);
                devices.date = new Date().toISOString().slice(0,10);

                devices.save();
                tanks.devices.push(devices);
                tanks.save();
                res.status(200).json(tanks);
        },
        AddTankByStation: async (req, res, next) => {
                const stations = await station.findById(req.body.id);
                const tanks = new tank(req.body.tank);
                var jogtables = [];
                jogtables = req.body.jogTables;

                jogtables.forEach(jogtablesadd => {
                        jogtablesadd = new jogTables(jogtablesadd);
                        jogtablesadd.save();
                        tanks.jogTables.push(jogtablesadd);
                });
                tanks.save();
                stations.tanks.push(tanks);
                stations.save();
                res.status(200).json(stations);
        },

        getTaknsByStation: async (req, res, next) => {
                const stations = await station.findById(req.params.stationId).populate("tanks");
                res.status(200).json(stations);
        },


        getLastHistoriqueByStation: async (req, res, next) => {
                let lastValue=0;
                let result = [];
                let stations = await station.findById(req.params.stationId).populate({
                        path: 'tanks',
                        model: 'tank', 
                        populate: [{
                                path: 'jogTables',
                                model: 'jogTables'
                            },
                            {
                                path: 'historiqueTank',
                                model: 'historiqueTank'
                            }]
                });
                let tanks = stations.tanks;
                tanks.forEach(async (element) => {
                        if(element.historiqueTank.pop()!=undefined){
                                
                      result.push(
                                {
                                        id: element.id,
                                        type : element.type,
                                        capacity:element.capacity,
                                        lastValue : element.historiqueTank.pop()
                                }
                        )
                }
                else{
                        result.push(
                                {
                                        id: element.id,
                                        type : element.type,
                                        capacity:element.capacity,
                                        lastValue : {value:0,date:null}
                                }
                        )

                }
               
                });
                res.status(200).json(result);
        },
        Notification: async (req, res) => {
                const subsciption =req.body;
                res.status(201).json()
            const payload =JSON.stringify({title :req.body.title,message:req.body.message});
            webpush.sendNotification(subsciption,payload).catch(err=>{console.log(err)})
             
            }
}
