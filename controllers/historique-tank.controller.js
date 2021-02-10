const router = require('express-promise-router')();
const HistorqiuetankService = require('../services/historique-tank.service');
const auth = require('../middleware/auth');


router.route('/')
    .get(HistorqiuetankService.getAll)
    .post(HistorqiuetankService.newhistoriqueTank)
    .delete(HistorqiuetankService.deleteAll);
    
router.route('/getListHitorique')
    .get(HistorqiuetankService.getLastHistoriqueByListTank)

router.route('/:historiqueTankId')
    .get(HistorqiuetankService.gethistoriqueTank)
    .patch(HistorqiuetankService.updatehistoriqueTank)
    .put(HistorqiuetankService.updatehistoriqueTank)
    .delete(HistorqiuetankService.deletehistoriqueTank)

router.route('/addhistorique/:idTank')
    .post(HistorqiuetankService.AddHistoriqueToTank)

router.route('/getHistorique/:historiqueTankId')
    .get(HistorqiuetankService.getHistoriqueByTank)

router.route('/getLastHistorique/:historiqueTankId')
    .get(HistorqiuetankService.getLastHistoriqueByTank)


module.exports = router;

