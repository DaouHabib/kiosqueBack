const router = require('express-promise-router')();
const pompeService = require('../services/pompe.service');
const auth = require('../middleware/auth');


router.route('/')
    .get(pompeService.getAll)
    .post(pompeService.newpompe)
    .delete(pompeService.deleteAll);

router.route('/:pompeId')
    .get(pompeService.getpompe)
    .patch(pompeService.updatePompe)
    .put(pompeService.updatePompe)
    .delete(pompeService.deletePompe)

router.route('/addPompe')
    .post(pompeService.AddPompeByStation)

router.route('/getPompes/:stationId')
    .get(pompeService.getPompesByStation)

    router.route('/getvente/:stationId')
    .get(pompeService.getVentesOfyesterdaybyStation)

    
module.exports = router;

