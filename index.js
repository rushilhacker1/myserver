//REST API demo in Node.js
var express = require('express'); // requre the express framework
var app = express();
const fs = require('fs'); //require file system object
const path = require('path');



app.use(express.urlencoded({ extended: false }));
app.use(express.json());

// Endpoint to Get a list of users
app.get('/getUser', function(req, res) {
    fs.readFile(__dirname + "/NaN/" + "users.json", 'utf8', function(err, data) {
        console.log(data);
        res.end(data); // you can also use res.send()
    });
    //Step 1: Create a new user variable
});


//The addUser endpoint
app.post('/addUser', function(req, res) {

    var data = req.body;
    console.log(data);
    fs.appendFileSync(path.resolve(__dirname, +'/NaN' + '/users.json'), JSON.stringify(data));


    res.status(201).send('user created');
});



app.get('/:id', function(req, res) {
    // First retrieve existing user list
    fs.readFile(__dirname + "/NaN/" + "users.json", 'utf8', function(err, data) {
        var users = JSON.parse(data);
        var user = users["user" + req.params.id]
        console.log(user);
        res.end(JSON.stringify(user));
    });
})

// Create a server to listen at port 8080
var server = app.listen(8080, '0.0.0.0', () => {
    var host = server.address().address
    var port = server.address().port
    console.log("REST API demo app listening at http://%s:%s", host, port)
})