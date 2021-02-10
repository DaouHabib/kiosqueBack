
var cron = require('node-cron');
const Tank = require('../models/tank');
const Historique = require('../models/historique-tank');
module.exports = {
    
    launch: async (req, res, next) => {
        const tanks = await Tank.findById(req.params.id);

        const xxx = cron.schedule('*/10 * * * * *', () => {
            // save historique 
         let newValue = new Historique({
                value:9000,
                date: Date.now(),
            });
         newValue.save();
            console.log(newValue);
            // affect historique to Tanc
            tanks.historiqueTank.push(newValue);
            tanks.save();
            console.log(tanks);
            // res.status(200); 
        });
    },

}
