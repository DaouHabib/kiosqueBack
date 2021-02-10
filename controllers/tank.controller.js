const router = require('express-promise-router')();
const tankService = require('../services/tank.service');
const auth = require('../middleware/auth');


router.route('/')
    .get(tankService.getAll)
    .post(tankService.newTank)
    .delete(tankService.deleteAll);

router.route('/:tankId')
    .get(tankService.getTank)
    .patch(tankService.updateTank)
    .put(tankService.updateTank)
    .delete(tankService.deleteTank)

router.route('/device')
    .post(tankService.AddDevice)

router.route('/addTank')
    .post(tankService.AddTankByStation)

router.route('/getTanks/:stationId')
    .get(tankService.getTaknsByStation)

router.route('/getHistorique/:stationId')
    .get(tankService.getLastHistoriqueByStation)


module.exports = router;

