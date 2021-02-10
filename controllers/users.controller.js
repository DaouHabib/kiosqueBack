const router = require('express-promise-router')();
const usersService = require('../services/users.service');
const { validateParam } = require('../helpers/utility');
const schemas = require('../helpers/validations');
const auth = require('../middleware/auth');
const { User } = require('../models/user');
const multer = require('multer');
const GridFsStorage = require('multer-gridfs-storage');
const Grid = require('gridfs-stream');
const nodemailer = require('nodemailer');
const jwt = require('jsonwebtoken');
const { JWT_SECRET } = require('../config/config');
const bcrypt = require('bcryptjs')
const  UserList  = require('../models/user');
const {pagination} = require('../helpers/utility')


router.get('/me', auth, async (req, res) => {
    const user = await User.findById(req.user._id).select('-password');
    res.send(user);
})
router.get('/pagination/:page/:limit', pagination(User), (req, res) => {
    res.json(res.pagination);
})
router.route('/')
    .get(usersService.getAll)
    .post(usersService.newUser)
    .delete(usersService.deleteAll)

router.route('/:userId')
    .get(usersService.getUser)
    .patch(usersService.updateUser)
    .put(usersService.updateUser)
    .delete(usersService.deleteUser)

router.route('/Employee/:stationId')
    .get(usersService.getEmployeeByStation)
    .post(usersService.addEmployeeToStation)

router.route('/Employee')
    .post(usersService.addEmployeeToStation)


router.route('/sendEmail')
    .post(usersService.sendemail)

module.exports = router;

