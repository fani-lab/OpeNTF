const fs = require('fs');
const path = require('path');

const outputDirectory = '../output';

function processDirectory(directory) {
  console.log(`Processing directory: ${directory}`);

  const processFiles = new Promise((resolve, reject) => {
    fs.readdir(directory, (err, files) => {
      if (err) {
        reject(`Error reading directory ${directory}: ${err}`);
        return;
      }

      // Exclude model_data.json file from the list of JSON files
      const jsonFiles = files.filter(file => path.extname(file) === '.json' && file !== 'model_data.json');
      console.log(`Found JSON files: ${jsonFiles}`);

      const filePromises = jsonFiles.map((file) => {
        const filePath = path.join(directory, file);

        return new Promise((resolveFile) => {
          fs.readFile(filePath, 'utf8', (err, fileContent) => {
            if (err) {
              console.error(`Error reading file ${filePath}`, err);
              resolveFile({ error: err });
              return;
            }

            try {
              const jsonData = JSON.parse(fileContent);

              if (Array.isArray(jsonData)) {
                const columnValues = jsonData.map((row) => row.mean);
                const fileName = path.parse(file).name;
                resolveFile({ fileName, columnValues });
              } else {
                console.error(`Error: JSON data in ${filePath} is not an array.`);
                resolveFile({ error: 'Invalid JSON data' });
              }
            } catch (error) {
              console.error(`Error parsing JSON in file ${filePath}`, error);
              resolveFile({ error });
            }
          });
        });
      });

      Promise.all(filePromises).then((results) => {
        const data = [];
        const errors = [];

        results.forEach((result) => {
          if (result.error) {
            errors.push(result.error);
          } else {
            data.push([result.fileName, ...result.columnValues]);
          }
        });

        if (data.length > 0) {
          const outputFileName = path.join(directory, 'model_data.json');
          const newJsonTable = {
            model_data: data
          };

          fs.writeFile(outputFileName, JSON.stringify(newJsonTable, null, 2), (err) => {
            if (err) {
              reject(`Error writing to file ${outputFileName}: ${err}`);
              return;
            }
            console.log('New JSON table created successfully:', outputFileName);

            if (errors.length > 0) {
              console.error('Errors occurred during processing:');
              errors.forEach((error) => {
                console.error(error);
              });
            }

            resolve();
          });
        } else {
          console.error('Error: No valid JSON files found.');
          resolve(); // Resolve even if no valid JSON files found
        }
      });
    });
  });

  return processFiles;
}

fs.readdir(outputDirectory, (err, files) => {
  if (err) {
    console.error('Error reading output directory', err);
    return;
  }

  console.log(`Found directories: ${files}`);

  const directoryPromises = files.map((file) => {
    const filePath = path.join(outputDirectory, file);
    return new Promise((resolveDir) => {
      fs.stat(filePath, (err, stats) => {
        if (err) {
          console.error(`Error reading file stats ${filePath}`, err);
          resolveDir();
          return;
        }

        if (stats.isDirectory()) {
          console.log(`Processing directory: ${filePath}`);
          processDirectory(filePath).then(resolveDir).catch(resolveDir);
        } else {
          resolveDir();
        }
      });
    });
  });

  Promise.all(directoryPromises).then(() => {
    console.log('All directories processed.');
  });
});
