const router = require('express-promise-router')();
const pistoletService = require('../services/pistolet.service');
const auth = require('../middleware/auth');


router.route('/')
    .get(pistoletService.getAll)
    .post(pistoletService.newPistolet)
    .delete(pistoletService.deleteAll);

router.route('/:pistoletId')
    .get(pistoletService.getPistolet)
    .patch(pistoletService.updatePistolet)
    .put(pistoletService.updatePistolet)
    .delete(pistoletService.deletePistolet)

router.route('/addPistolet')
    .post(pistoletService.AddPistoletByPompe)

router.route('/getPistolets/:pompeId')
    .get(pistoletService.getPistoletsByPompe)

module.exports = router;

