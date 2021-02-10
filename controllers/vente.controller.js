const router = require('express-promise-router')();
const venteService = require('../services/vente.service');
const auth = require('../middleware/auth');


router.route('/')
    .get(venteService.getAll)
    .post(venteService.newvente)
    .delete(venteService.deleteAll);

router.route('/:venteId')
    .get(venteService.getvente)
    .patch(venteService.updatevente)
    .put(venteService.updatevente)
    .delete(venteService.deletevente)

router.route('/getYesterdayVente/:id')
.get(venteService.getVenteYesterday)

router.route('/last24/:stationId')
.get(venteService.getVentebyidStation)

router.route('/getVenteByposte/:stationId')
.get(venteService.getVentebyPoste)


router.route('/getVenteBytype/:stationId')
.get(venteService.getVentebyType)


module.exports = router;

