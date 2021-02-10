const router = require('express-promise-router')();
const ravitaillementService = require('../services/ravitaillement.service');
const auth = require('../middleware/auth');


router.route('/')
    .get(ravitaillementService.getAll)
    .post(ravitaillementService.newravitaillement)
    .delete(ravitaillementService.deleteAll);

router.route('/:ravitaillementId')
    .get(ravitaillementService.getravitaillement)
    .patch(ravitaillementService.updateravitaillement)
    .put(ravitaillementService.updateravitaillement)
    .delete(ravitaillementService.deleteravitaillement)
    
router.route('/bytank/:id')
.get(ravitaillementService.getBytank)
.post(ravitaillementService.addBytank)
module.exports = router;

