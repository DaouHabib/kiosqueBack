const vente = require('../models/vente');

module.exports = {


    getAll: async (req, res, next) => {
        const ventes = await vente.find({});
        res.status(200).json(ventes);
    },

    newvente: async (req, res, next) => {
        ventes = new vente(req.body);
        await ventes.save();
        res.status(200).json(ventes);
    },

    getvente: async (req, res, next) => {
        const ventes = await vente.findById(req.params.venteId);
        res.status(200).json(ventes);
    },

    // PATCH || PUT
    updatevente: async (req, res, next) => {
        const newvente = req.body;
        const ventes = await vente.findByIdAndUpdate(req.params.venteId, newvente);
        res.status(200).json('success');
    },

    deletevente: async (req, res, next) => {
        const ventes = await vente.findOneAndDelete(req.params.venteId).exec(function(err, item) {
            if (err) {
                return res.json({success: false, msg: 'Cannot remove item'});
            }       
            if (!item) {
                return res.status(404).json({success: false, msg: 'Vente not found'});
            }  
            res.json({success: true, msg: 'Vente deleted.'});
        });
    },
    
    deleteAll: async (req, res, next) => {
        const ventes = await vente.deleteMany();
        res.status(200).json('success');
    },

    getVenteYesterday: async(req,res,next)=> {

        const date =new Date().toISOString().slice(0,10);
        const day =date.slice(8,10);
        res.status(200).json('success');

    },
    getVentebyidStation: async(req,res,next)=> {
     
        const ventes = await vente.find({stationId:req.params.stationId});
        let v = ventes.sort().shift();
        console.log(v);
        res.status(200).json({totalEssence:v.totalEssence,totalDisel:v.totalDisel});

    },
    getVentebyPoste: async(req,res,next)=> {
        let resltPost1=[];
        let resltPost2=[];
        let resltPost3=[];
        let resltDate=[];

        const ventes = await vente.find({stationId:req.params.stationId});

        ventes.forEach(data=>{
            if(data){
                resltDate.push(data.date);
            resltPost1.push(data.totalpost1);
            resltPost2.push(data.totalpost2);
            resltPost3.push(data.totalpost3);
        }
        })

        res.status(200).json({resltPost1 : resltPost1  , resltPost2 : resltPost2 ,resltPost3 :resltPost3,resltDate:resltDate});
    },
    getVentebyType: async(req,res,next)=> {
       
        let TotalEssence=0;
        let  TotalGasoil =0;
        let ArrayTotalEssence=[];
        let ArrayTotalGasoil =[];
        let resltDate=[];
        const ventes = await vente.find({stationId:req.params.stationId});

        ventes.forEach(data=>{
            if(data){
                data.totalEssence.forEach(element => {
                   TotalEssence+=element;
                 });
                 data.totalDisel.forEach(element => {
                    TotalGasoil +=element;
                 });
         
        }
        resltDate.push(data.date);

        ArrayTotalEssence.push(TotalEssence);
        ArrayTotalGasoil.push(TotalGasoil);
        })

        res.status(200).json({totalEssence : ArrayTotalEssence  , totalGasoil : ArrayTotalGasoil ,resltDate:resltDate});
    }



}
