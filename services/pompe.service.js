const pompe = require('../models/pompe');
const station = require('../models/station-Service');
const pistolet = require('../models/pistolet');
const { string } = require('joi');

module.exports = {


    getAll: async (req, res, next) => {
        const pompes = await pompe.find({});
        res.status(200).json(pompes);
    },

    newpompe: async (req, res, next) => {
        pompes = new pompe(req.body);
        await pompes.save();
        res.status(200).json(pompes);
    },

    getpompe: async (req, res, next) => {
        const pompes = await pompe.findById(req.params.pompeId);
        res.status(200).json(pompes);
    },

    // PATCH || PUT
    updatePompe: async (req, res, next) => {
        const newpompe = req.body;
        const pompes = await pompe.findByIdAndUpdate(req.params.pompeId, newpompe);
        res.status(200).json('success');
    },

    deletePompe: async (req, res, next) => {
        const pompess = await pompe.findById(req.params.pompeId);
        pompess.pistolets.forEach(res=>{
            pistolet.findByIdAndDelete(res._id);   
        })
        const pompes = await pompe.findOneAndDelete(req.params.pompeId).exec(function(err, item) {
            if (err) {
                return res.json({success: false, msg: 'Cannot remove item'});
            }       
            if (!item) {
                return res.status(404).json({success: false, msg: 'Pompe not found'});
            }  
            res.json({success: true, msg: 'Pompe deleted.'});
        });
    },
    
    deleteAll: async (req, res, next) => {
        const pompes = await pompe.deleteMany();
        res.status(200).json('success');
    },
    AddPompeByStation: async (req, res, next) => {
        const stations = await station.findById(req.body.id);
        const pompes = new pompe(req.body.pompe);
        pompes.save();
        stations.pompes.push(pompes);
        stations.save();
        res.status(200).json(stations);
},

getPompesByStation: async (req, res, next) => {
        const stations = await station.findById(req.params.stationId).populate("pompes");
        res.status(200).json(stations);
},
getVentesOfyesterdaybyStation: async (req, res, next) => {
    let pistolets=[];
    let post1 = [{type:String,value:Number}];
    let post2 = [{type:String,value:Number}];
    let post3 = [{type:String,value:Number}];
    let tEssence =0;
    let Tdisel = 0;
    let types=[{total :tEssence, type:"ESSENCE"},{total :Tdisel, type:"DISEL"}];
    let postes=[{typePoste:"Post1",totalPost:post1},{typePoste:"Post2",totalPost:post2},{typePoste:"Post3",totalPost:post3}]
    console.log (req);

    let stations = await station.findById(req.params.stationId).populate({
        path: 'pompes',
        model: 'pompe', 
        populate: [{
                path: 'pistolets',
                model: 'pistolet',
                populate:[{
                    path: 'historiqueIndexes',
                    model: 'historiqueIndex',
                }]
            },
           ]
});
    stations.pompes.forEach(res=>{
        res.pistolets.forEach(data=>{
            pistolets.push(data);
        })
    })

    pistolets.forEach(pis =>{
    if(pis.historiqueIndexes.length !=0 && pis.historiqueIndexes){
            console.log(pis.historiqueIndexes.length);
            pis.historiqueIndexes= pis.historiqueIndexes.slice(0).slice(-3)
}

pis.historiqueIndexes.forEach(element => {
    let total;
    const date =new Date(element.date).toISOString().slice(0,10);
    postes.forEach(p=>{
    types.forEach(t=>{
        if(element.typePoste==p.typePoste && pis.type == t.type){
       t.total+=element.indexfin-element.indexdebut
       p.totalPost.push({type:t.type,value:element.indexfin-element.indexdebut});
    }

    
    })

    })
})

})

    res.status(200).json({type : types ,poste : postes});
},


getVentesyesterday: async (req, res, next) => {
    let pistolets=[];
    let Totalpost1 = 0;
    let Totalpost2 = 0;
    let Totalpost3 = 0;
    let Totalessence = [];
    let TotalDisel = [];
    let TotalEssence1=0 ;
    let TotalDisel1=0 ;
    let TotalEssence2=0 ;
    let TotalDisel2=0 ;
    let TotalEssence3=0 ;
    let TotalDisel3=0 ;
    let date;
    console.log (req);
    //GET  station by Id with pompes pistolets and historiqueindexes
    let stations = await station.findById(req).populate({
        path: 'pompes',
        model: 'pompe', 
        populate: [{
                path: 'pistolets',
                model: 'pistolet',
                populate:[{
                    path: 'historiqueIndexes',
                    model: 'historiqueIndex',
                }]
            },
           ]
});

//get all Pitolets by kiosque   
    stations.pompes.forEach(res=>{
        res.pistolets.forEach(data=>{
            pistolets.push(data);
        })
    })
//get last Values of historique index By pistolet 
    pistolets.forEach(pis =>{
    if(pis.historiqueIndexes.length !=0 && pis.historiqueIndexes){
            pis.historiqueIndexes= pis.historiqueIndexes.slice(0).slice(-3)
}
console.log(pis.historiqueIndexes);
pis.historiqueIndexes.forEach(element => {
    let total;
    const date =new Date(element.date).toISOString().slice(0,10);
    if(element.typePoste=="Post1" && pis.type == "ESSENCE"){
        total=element.indexfin-element.indexdebut;
        TotalEssence1+=total;

    }
    if(element.typePoste=="Post2" && pis.type=="ESSENCE"){
        total=element.indexfin-element.indexdebut;

        TotalEssence2+=total;

    }
    if(element.typePoste=="Post3" && pis.type=="ESSENCE"){
        total=element.indexfin-element.indexdebut;

        TotalEssence3+=total;

    }

    if(element.typePoste=="Post1" && pis.type == "DISEL"){
        total=element.indexfin-element.indexdebut;

       
        TotalDisel1+=total;
    }
    if(element.typePoste=="Post2" && pis.type=="DISEL"){
        total=element.indexfin-element.indexdebut;
        
        TotalDisel2+=total;

    }
    if(element.typePoste=="Post3" && pis.type=="DISEL"){
        total=element.indexfin-element.indexdebut;
        TotalDisel3+=total;
    }
});


})

Totalessence.push(TotalEssence1*1.975);
Totalessence.push(TotalEssence2*1.975);
Totalessence.push(TotalEssence3*1.975);
TotalDisel.push(TotalDisel1*1.750);
TotalDisel.push(TotalDisel2*1.750);
TotalDisel.push(TotalDisel3*1.750);
Totalpost1 = (TotalEssence1*1.975)+(TotalDisel1*1.750);
Totalpost2 = (TotalEssence2*1.975)+(TotalDisel2*1.750);
Totalpost3 = (TotalEssence3*1.975)+ (TotalDisel3*1.750);
dateNow=new Date().toISOString().slice(1, 10);
return{  totalEssence: Totalessence,totalDisel : TotalDisel ,date :dateNow ,stationId:req , 
    totalpost1 :Totalpost1,totalpost2:Totalpost2 ,totalpost3 :Totalpost3};

},


}
