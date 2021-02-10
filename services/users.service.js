const { User } = require('../models/user');
const { hashedPwd } = require('../helpers/utility');
const stationService = require('../models/station-Service');
const tank = require('../models/tank');
const device = require('../models/device');
const jogTables = require('../models/jogTables');
const serviceStation = require('../services/station.Service')
const JSON = require('circular-json');
const nodemailer = require('nodemailer');
const jwt = require('jsonwebtoken');
const { JWT_SECRET } = require('../config/config');
const bcrypt = require('bcryptjs');

require('dotenv').config();


module.exports = {

        getAll: async (req, res, next) => {
                const users = await User.find({});
                res.status(200).json(users);
        },

        newUser: async (req, res, next) => {


                let user = await User.findOne({ email: req.body.email });
                if (user) return res.status(400).send('user already registred');

                user = new User(req.body);
                user.creation_dt = Date.now();
                user.password = await hashedPwd(req.body.password);
                user.imageUrl = req.body.imageUrl;
                await user.save();

                res.status(201).json((user));
        },

        getUser: async (req, res, next) => {

                const user = await User.findById(req.params.userId);
                res.status(200).json(user);
        },

        // PATCH || PUT
        updateUser: async (req, res, next) => {
                const newUSer = req.body
                if(newUSer.password){
                const userCheck = await User.findById(req.params.userId);
                const validpwd = await bcrypt.compare(newUSer.password, userCheck.password);
                if (!validpwd){
                        newUSer.password = await hashedPwd(newUSer.password);                }
                const user = await User.findByIdAndUpdate(req.params.userId, newUSer);
        }
        else{                const user = await User.findByIdAndUpdate(req.params.userId, newUSer);
        }
                res.status(200).json('success');
        },

        deleteUser: async (req, res, next) => {
                const user = await User.findOneAndDelete(req.params.userId).exec(function(err, item) {
                        if (err) {
                            return res.json({success: false, msg: 'Cannot remove item'});
                        }       
                        if (!item) {
                            return res.status(404).json({success: false, msg: 'User not found'});
                        }  
                        res.json({success: true, msg: 'User deleted.'});
                    });
        },
        deleteAll: async (req, res, next) => {
                const users = await User.deleteMany();
                res.status(200).json('success');
        },

   
        addEmployeeToStation: async (req, res, next) => {
                let user = await User.findOne({ email: req.body.email });
                if (user) return res.status(400).send('user already registred');
                var s =[] ;
                user = new User(req.body);
                user.creation_dt = Date.now();
                user.password = await hashedPwd(req.body.password);
                user.imageUrl = req.body.imageUrl;
                user.Role="EMPLOYEE";
                const station = await stationService.findById(req.params.stationId)
                console.log(station);
                user.stationServices.push(station);
                await user.save();

                res.status(201).json((user));
        },
       
        
        getEmployeeByStation: async (req, res, next) => {
               
               var ListUsers=[];
                const user = await User.find({Role:"EMPLOYEE"}).populate("stationServices");
                user.forEach(data=>{
                        data.stationServices.forEach(res=>{
                                if (res._id == req.params.stationId){
                                        ListUsers.push(data);
                                        
                                }

                        })
                })



                res.status(200).json(ListUsers);
        },

        AddFullUser: async (req, res, next) => {
                let user = await User.findOne({ email: req.body.user.email });
                if (user) return res.status(400).send('user already registred');
                var tanks = [];
                var jogtables = [];
                user = new User(req.body.user);
                user.creation_dt = Date.now();
                user.password = await hashedPwd(user.password);
                const sts = new stationService(req.body.stationServices);
                jogtables = req.body.jogTables;
                tanks = req.body.tank;
                tanks.forEach(tankadd => {
                        tankadd = new tank(tankadd);
                        jogtables.forEach(jogtablesadd => {
                                jogtablesadd = new jogTables(jogtablesadd);
                                jogtablesadd.save();
                                tankadd.jogTables.push(jogtablesadd);

                        });

                        tankadd.save();
                        sts.tanks.push(tankadd);
                });
                sts.save();
                user.stationServices.push(sts);
                user.save();
                res.status(200).json(user);
        },

        sendemail: async (req, res, next) => {
                let user = await User.findOne({ email:req.body.email });
                if (!user) return res.status(400).send('user not registred');
               
                const token = jwt.sign({
                        _id:user._id,firstName:user.firstName,lastName: user.lastName,Role:user.Role,
                        iss: 'habibdaou',
                        sub: user.id,
                        iat: new Date().getTime(),
                        exp: new Date().setDate(new Date().getDate() + 1)
                    }, JWT_SECRET);


                let transporter = nodemailer.createTransport({
                        port:587,
                        secure:false,
                        service: 'gmail',
                        auth: {
                                user: process.env.EMAIL,
                                pass: process.env.PASSWORD,
                        }
                });
                let mailOptions = {

                        from: 'nolivetg@gmail.com',
                        to: user.email,
                        subject: 'test works',
                        text: 'welcom to ower application if you need to navigate to http://localhost:4200/authentication/forgotPassword/'+token
                }

                transporter.sendMail(mailOptions, function (err, data) {
                        if (err) {
                                console.log(err);
                                console.log(process.env.EMAIL);
                                res.status(400).json({error:"Error"});
                        } else {
                                console.log('email sent!! ');
                                res.status(200).json({Sucess:"sucess"});
                        }

                })


        },
      
}
