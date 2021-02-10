const express = require('express');
const logger = require('morgan');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const path = require('path');
const methodOverride = require('method-override');
const dbConfig = require('./config/config.js');
const auth = require('./controllers/auth.controller');
const users = require('./controllers/users.controller');
const stations = require('./controllers/station.controller');
const uploads = require('./controllers/uploads.controller');
const tanks = require('./controllers/tank.controller');
const devices = require('./controllers/device.controller');
const jogTables = require('./controllers/jogTables.controller');
const pompes = require('./controllers/pompe.controller');
const pistolet = require('./controllers/pistolet.controller');
const historiqueTank = require('./controllers/historique-tank.controller');
const historiqueIndexes = require('./controllers/historique_indexes.controller');
const vente = require('./controllers/vente.controller');
const cronController = require('./controllers/cron.controller');
const subscriberController = require('./controllers/subscriber.controller');
const ravitaillmentController = require('./controllers/ravitaillement.controller');
const tenserController = require('./services/tenserVente');
const historiqueTanks = require('./models/historique-tank');

var request = require('request');

var cron = require('node-cron');
var cors = require('cors');
var http = require('http');
// Connection to DataBase
mongoose.connect(dbConfig.url, { useNewUrlParser: true })
    .then(() => {
        console.log("Successfully connected to the database");
    }).catch(err => {
        console.log('Could not connect to the database. Exiting now...', err);
        process.exit();
    });

var app = express();
app.use(express.static(path.join(__dirname,"client")));
// CORS
app.use(function (req, res, next) {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
    next();
});

// Midleware
mongoose.set('useCreateIndex', true) // ?
app.use(logger('dev'));
app.use(bodyParser.json());
app.use(cors());
app.use(methodOverride('_method'));


// cron
// var task = cron.schedule('*/10 * * * * *', () => {
//     var d = new historiqueTanks({
//         value: Math.floor(Math.random() * 7801),
//         date: Date.now(),

//     })
//     request.post(
//         'http://localhost:3000/historiqueTank/addhistorique/5ef916413cb6b5165869ed32',
//         { json: d },
//         function (error, response, body) {
//             if (!error && response.statusCode == 200) {
//                 console.log(body);
//             }
//         }
//     );

// })
// task.start()


// APIS
app.use('/auth', auth);
app.use('/users', users);
app.use('/stations', stations);
app.use('/uploads', uploads);
app.use('/tanks', tanks);
app.use('/devices', devices);
app.use('/jogTables', jogTables);
app.use('/pompes', pompes);
app.use('/pistolets', pistolet);
app.use('/historiqueTank', historiqueTank);
app.use('/historiqueIndexe', historiqueIndexes);
app.use('/ventes', vente);
app.use('/cron', cronController);
app.use('/subscribe', subscriberController);
app.use('/ravitaillment', ravitaillmentController);
app.use('/tenser',tenserController );



// catch 404 Error anf foward to erroHandler
app.use((req, res, next) => {
    const err = new Error('Not found');
    err.status = 404;
    next(err)
});

// Error handler
app.use((err, req, res, next) => {
    const error = app.get('env') === 'developement' ? err : {};
    const status = err.status || 500;
    // response to client
    res.status(status).json({
        error: {
            message: error.message
        }
    })
    console.log(err)
});


//socket.io
const server = http.createServer(app);
const socketIo = require("socket.io");
const io = socketIo.listen(server);

var idTank ="";

io.on('connection', (socket) => {

    socket.on('msg', (data) => {
        this.idTank = data.msg;
    });
    console.log('a user connected'), setInterval(
        () => getApiAndEmit(socket,this.idTank),
        6000000
    );
    socket.on('disconnect', () => {
        console.log('user disconnected');
    });



});
const getApiAndEmit = async (socket,id) => {
    try {
        var res;
        request.get(
            'http://localhost:3000/historiqueTank/getHistorique/'+id,
            {},
            function (error, response, body) {
                if (!error && response.statusCode == 200) {
                    socket.emit("From db", body);
                    console.log(body);
                }
            }

        );


    } catch (error) {
        console.error(`Error: ${error.code}`);
    }
};

// start the server
const port = app.get('port') || 3000;
server.listen(port, () => console.log('server in listening on port 3000'));