const router = require('express-promise-router')();
const jogTableService = require('../services/jogTables.service');
const auth = require('../middleware/auth');


router.route('/')
    .get(jogTableService.getAll)
    .post(jogTableService.newjogTable)
    .delete(jogTableService.deleteAll);

router.route('/:jogId')
    .get(jogTableService.getjogTable)
    .delete(jogTableService.deletedevice)






module.exports = router;

