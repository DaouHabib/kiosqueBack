const router = require('express-promise-router')();
const deviceService = require('../services/device.service');
const auth = require('../middleware/auth');


router.route('/')
    .get(deviceService.getAll)
    .post(deviceService.newdevice)
    .delete(deviceService.deleteAll);

router.route('/:deviceId')
    .get(deviceService.getdevice)
    .patch(deviceService.updateDevice)
    .put(deviceService.updateDevice)
    .delete(deviceService.deletedevice)

module.exports = router;

