const router = require('express-promise-router')();
const stationService = require('../services/station.Service');
const auth = require('../middleware/auth');

router.route('/')
    .get(stationService.getAll)
    .post(stationService.newstation)
    .delete(stationService.deleteAll);

router.route('/:stationId')
    .get(stationService.getStation)
    .patch(stationService.updateStation)
    .put(stationService.updateStation)
    .delete(stationService.deleteStation)

router.route('/Station')
    .post(stationService.AddStationToUser)

router.route('/getStations/:userId')
    .get(stationService.getallStationByUser)

router.route('/venteStation/:id')
    .get(stationService.addVentes)


module.exports = router;

