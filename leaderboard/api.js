const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();

// Enable Cross Origin Resource Sharing to all origins by default
app.use(cors());

// Serve the static files in the /public directory
app.use(express.static('public'));

// An api endpoint that returns all available dataset directories
app.get('/datasets', (req, res) => {
    fs.readdir('../output', (err, files) => {
        if (err) {
            res.status(500).send('Error reading output directory');
        } else {
            // Filter out any non-directory files
            const datasets = files.filter(file => fs.statSync('../output/'+file).isDirectory());
            res.json(datasets);
        }
    });
});

// An api endpoint that returns a specific dataset's model data
app.get('/:dataset/model_data.json', (req, res) => {
    const dataset = req.params.dataset;

    // Create a path to the file
    let filePath = path.join(__dirname, '..', 'output', dataset, 'model_data.json');

    // Check if file exists before sending
    if (fs.existsSync(filePath)) {
        res.sendFile(filePath);
    } else {
        res.status(404).send('File not found');
    }
});

const port = 5000;
app.listen(port, () => console.log(`Server running on port ${port}`));
