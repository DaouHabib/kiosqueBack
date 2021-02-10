const router = require('express-promise-router')();
const cronService = require('../services/cron.service');

router.route('/:id')
    .get(cronService.launch)
module.exports = router;

