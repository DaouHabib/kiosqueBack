const router = require('express-promise-router')();
const historiqueIndexes = require('../services/historique_indexes.service');
const auth = require('../middleware/auth');


router.route('/')
    .get(historiqueIndexes.getAll)
    .post(historiqueIndexes.newhistoriqueIndexe)
    .delete(historiqueIndexes.deleteAll);
    
router.route('/:historiqueIndexeId')
    .get(historiqueIndexes.gethistoriqueIndexe)
    .patch(historiqueIndexes.updatehistoriqueIndexe)
    .put(historiqueIndexes.updatehistoriqueIndexe)
    .delete(historiqueIndexes.deletehistoriqueIndexe)

router.route('/addIndexe/:idpistolet')
    .post(historiqueIndexes.AddHistoriqueToTank)

router.route('/getHistorique/:historiqueIndexeId')
    .get(historiqueIndexes.getHistoriqueByTank)

router.route('/getVentes/:idpistolet')
    .get(historiqueIndexes.getVentesOfyesterday)

module.exports = router;

