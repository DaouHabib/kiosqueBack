const jwt = require('jsonwebtoken');

module.exports = function (req, res, next) {
    const token = req.header('x-auth-token');
    if (!token) res.status(401).send('access denied . No Token Provided.');
    try {
        const decodedPayload = jwt.verify(token, "jwtPrivateKey");
        req.user = decodedPayload;
        next();
    }
    catch (ex) {
        res.status(400).send('invalid token.')
    }
}

