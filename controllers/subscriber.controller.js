const router = require('express-promise-router')();
const alertService = require('../services/alert.service')


router.route('/')
    .get(alertService.getAll)
    .post(alertService.Notification)

router.route('/:idUser')
    .get(alertService.getbyIduser)



module.exports = router;

