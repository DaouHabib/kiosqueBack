const router = require('express-promise-router')();
const authService = require('../services/auth.service');

router.route('/login' )
    .post(authService.login)

module.exports = router;