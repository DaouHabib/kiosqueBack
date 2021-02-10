const router = require('express-promise-router')();
const prixService = require('../services/prix.service');
const auth = require('../middleware/auth');


router.route('/')
    .get(prixService.getAll)
    .post(prixService.newprix)
    .delete(prixService.deleteAll);

router.route('/:prixId')
    .get(prixService.getprix)
    .patch(prixService.updateprix)
    .put(prixService.updateprix)
    .delete(prixService.deleteprix)

module.exports = router;

