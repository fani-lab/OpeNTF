const fs = require('fs');
const path = require('path');

function csvToJson(csv) {
    let lines = csv.split("\n");
    let headers = ["FIELD1", "mean"];
    let json = [];

    for (let i = 1; i < lines.length; i++) {
        let obj = {};
        let currentline = lines[i].split(",");

        // Ignore empty lines
        if (currentline.length < 2) continue;

        for (let j = 0; j < headers.length; j++) {
            obj[headers[j].trim()] = currentline[j].trim();
        }

        json.push(obj);
    }

    return JSON.stringify(json, null, 2); // pretty print
}

function processModelDirectory(directory, datasetDirectory, modelName) {
    let unknown3 = getFirstSubDirectory(directory);
    let csvFilePath;
    
    if (unknown3) {
        csvFilePath = path.join(directory, unknown3, 'test.pred.eval.mean.csv');
    } else {
        csvFilePath = path.join(directory, 'test.pred.eval.mean.csv');
    }

    if (fs.existsSync(csvFilePath)) {
        console.log('Converting CSV to JSON:', csvFilePath);
        let csv = fs.readFileSync(csvFilePath, 'utf8');
        let json = csvToJson(csv);

        // Output json file to the dataset directory
        let outputFilePath = path.join(datasetDirectory, `${modelName}.json`);
        console.log('Writing JSON output to:', outputFilePath);
        fs.writeFileSync(outputFilePath, json);
    } else {
        console.log('CSV file not found:', csvFilePath);
    }
}

function getFirstSubDirectory(dirPath) {
    return fs.readdirSync(dirPath).find(file => {
        let fullPath = path.join(dirPath, file);
        return fs.statSync(fullPath).isDirectory();
    });
}

function processDirectory(directory) {
    fs.readdir(directory, (err, datasets) => {
        if (err) {
            console.error('Error reading directory:', directory, err);
            return;
        }

        datasets.forEach(dataset => {
            let datasetDirPath = path.join(directory, dataset);
            if (fs.statSync(datasetDirPath).isDirectory()) { // check if it's a directory
                let unknown1 = getFirstSubDirectory(datasetDirPath);

                if (unknown1) {
                    let modelDirPath = path.join(datasetDirPath, unknown1);
                    let models = fs.readdirSync(modelDirPath);

                    models.forEach(model => {
                        let unknown2DirPath = path.join(modelDirPath, model);
                        let unknown2 = getFirstSubDirectory(unknown2DirPath);

                        if (unknown2) {
                            let csvDirPath = path.join(unknown2DirPath, unknown2);
                            processModelDirectory(csvDirPath, datasetDirPath, model);
                        }
                    });
                }
            }
        });
    });
}

processDirectory('../output'); // start processing from the 'output' directory
