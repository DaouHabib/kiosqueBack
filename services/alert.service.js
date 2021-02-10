const webpush = require('web-push');
const privateKey= 'XAtPjgxcdQgfPc7-4uVG3LbajICCXQmHW3QpffHa3sc';
const publicKey= 'BAkmXAVNU7muOJ_ZUpD3ca5rzKr-CoaPusTzGQD96VuEeoYtCVrOXg8Ov3Wg0fVNm3Xo7DbIituY10sb7uh5LUg';
webpush.setVapidDetails("mailTo:test@test.com",publicKey,privateKey)
const Alert = require('../models/Alerts');

module.exports = {
    getAll: async (req, res) => {
      const  alerts= await Alert.find({});
      res.status(200).json(alerts);
     
    },

        Notification: async (req, res) => {
            const subsciption =req.body;
                const alerts =new Alert({title :req.body.title,message:req.body.message, date : Date.now(),idUser:req.body.idUser});
                alerts.save();
                res.status(201).json(alerts);
            const payload =JSON.stringify({title :req.body.title,message:req.body.message});
            webpush.sendNotification(subsciption,payload).catch(err=>{console.log(err)})
             
            },
            getbyIduser: async (req, res) => {
                const  alerts= await Alert.find({idUser:req.params.idUser});
                res.status(200).json(alerts);
               
              },
}
