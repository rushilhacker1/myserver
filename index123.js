const fs = require("fs");
const http = require("http");



const server = http.createServer((req, res) => {

        if (req.url == "/myself") {

            res.end("hi");

        } else if (req.url == "/EFSHDFHSDF") {

            res.end("hello");


        } else if (req.url == "/data") {

            fs.readFile('${_dirname}/data.json', "utf-8", (err, data) => {

                console.log(data);

            });
            res.end("url defined")

        } else {

            res.writeHead(404, { "Content-type": "text/html" });
            res.end("<h1>404 error. page doesn't exist </h1>");

        }
    }

);
server.listen(8080, () => {
    console.log("server started at port 8080");


});