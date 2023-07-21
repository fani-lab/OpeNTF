const fs = require('fs');
const path = require('path');

const jsonDirectory = 'json';
const outputFileName = 'model_data.json';
const data = [];

fs.readdir(jsonDirectory, (err, files) => {
  if (err) {
    console.error(err);
    return;
  }

  files.forEach((file) => {
    const filePath = path.join(jsonDirectory, file);

    fs.readFile(filePath, 'utf8', (err, fileContent) => {
      if (err) {
        console.error(filePath, err);
        return;
      }

      try {
        const jsonData = JSON.parse(fileContent);
        const columnValues = jsonData.map((row) => row.mean);
        const fileName = path.parse(file).name;
        data.push([fileName, ...columnValues]);

        if (data.length === files.length) {
          const newJsonTable = {
            model_data: data
          };

          fs.writeFile(outputFileName, JSON.stringify(newJsonTable, null, 2), (err) => {
            if (err) {
              console.error(err);
              return;
            }
            console.log('New JSON table created successfully:', outputFileName);
          });
        }
      } catch (error) {
        console.error(filePath, error);
      }
    });
  });
});
