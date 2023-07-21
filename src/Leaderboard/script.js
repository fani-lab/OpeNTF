colors = [
  "",
  "rgba(255, 0, 0, ",
  "rgba(0, 0, 255, ",
  "rgba(0, 255, 0, ",
  "rgba(255, 255, 0, ",
  "rgba(128, 0, 128, ",
  "rgba(255, 165, 0, ",
  "rgba(255, 192, 203, ",
  "rgba(0, 255, 255, ",
  "rgba(255, 0, 255, ",
  "rgba(0, 128, 128, ",
  "rgba(128, 128, 128, ",
  "rgba(128, 128, 0, ",
  "rgba(255, 215, 0, "
]



selectedMetricValue = "P";

function selectMetric(element, metric) {
  var menuItems = document.querySelectorAll('.menu li');
  menuItems.forEach(function(item) {
    item.classList.remove('selected');
  });
  element.classList.add('selected');
  selectedMetricValue = metric;
  for(var elem of document.querySelectorAll('.menustatic li')){
    if(elem.textContent.includes(selectedMetricValue)){
      if(selectedMetricValue == "P"){
        if(elem.textContent[0] == "P"){
          elem.classList.add('selected');
        }
      }
      else{
        elem.classList.add('selected');
      }
    }
    else{
      elem.classList.remove('selected')
    }
  }
  fetchData()
  .then(data => {
    populateTable(data);
    updateChartWithData(data.model_data, selectedMetricValue);
  });
}

function fetchData() {
  return fetch('./model_data.json')
    .then(response => {
      if (response.ok) {
        return response.json();
      } else {
        throw new Error('Error loading JSON data');
      }
    })
    .catch(error => {
      console.error('Error:', error);
    });
}

function updateChartWithData(modelData, metric) {
  var r = 0;
  if(metric == "P"){
    r = 0;
  }
  else if(metric == "Recall"){
    r = 3;
  }
  else if(metric == "nDCG"){
    r = 6;
  }
  else if(metric == "mAP"){
    r = 9;
  }
  else{
    r = 12
  }

  if(r!=12){
    var graph = {
      labels: ["", ...modelData.map(row => row[0]), ""],
      datasets: [
        {
          label: metric+"2",
          fillColor: colors[r+1]+"0.1)",
          strokeColor: colors[r+1]+" 1)",
          pointColor: colors[r+1]+" 1)",
          pointStrokeColor: "#202b33",
          pointHighlightStroke: "rgba(225,225,225,0.9)",
          data: [null, ...modelData.map(row => parseFloat(row[r+1]).toFixed(3)), null]
        },
        {
          label: metric+"5",
          fillColor: colors[r+2]+" 0.1)",
          strokeColor: colors[r+2]+" 1)",
          pointColor: colors[r+2]+" 1)",
          pointStrokeColor: "#202b33",
          pointHighlightStroke: "rgba(225,225,225,0.9)",
          data: [null, ...modelData.map(row => parseFloat(row[r+2]).toFixed(3)), null]
        },
        {
          label: metric+"10",
          fillColor: colors[r+3]+" 0.3)",
          strokeColor: colors[r+3]+" 1)",
          pointColor: colors[r+3]+" 1)",
          pointStrokeColor: "#202b33",
          pointHighlightStroke: "rgba(225,225,225,0.9)",
          data: [null, ...modelData.map(row => parseFloat(row[r+3]).toFixed(3)), null]
        }
      ]
    };
    var ctx = document.getElementById("graph").getContext("2d");
    window.myLineChart = new Chart(ctx).Line(graph, {
      pointDotRadius: 6,
      pointDotStrokeWidth: 2,
      datasetStrokeWidth: 3,
      scaleShowVerticalLines: false,
      scaleGridLineWidth: 2,
      scaleShowGridLines: true,
      scaleGridLineColor: "rgba(225, 255, 255, 0.02)",
      scaleOverride: true,
      scaleSteps: 10,
      scaleStepWidth: 0.01,
      scaleStartValue: 0,
      responsive: true,
      offsetgridlines: true,
      datasetFill: false
    });
  }
  else{
    var graph = {
      labels: ["", ...modelData.map(row => row[0]), ""],
      datasets: [
        {
          label: metric,
          fillColor: colors[r+1]+"0.1)",
          strokeColor: colors[r+1]+" 1)",
          pointColor: colors[r+1]+" 1)",
          pointStrokeColor: "#202b33",
          pointHighlightStroke: "rgba(225,225,225,0.9)",
          data: [null, ...modelData.map(row => parseFloat(row[r+1]).toFixed(3)), null]
        }
      ]
    };
    var ctx = document.getElementById("graph").getContext("2d");
    window.myLineChart = new Chart(ctx).Line(graph, {
      pointDotRadius: 6,
      pointDotStrokeWidth: 2,
      datasetStrokeWidth: 3,
      scaleShowVerticalLines: false,
      scaleGridLineWidth: 2,
      scaleShowGridLines: true,
      scaleGridLineColor: "rgba(225, 255, 255, 0.02)",
      scaleOverride: true,
      scaleSteps: 10,
      scaleStepWidth: 0.1,
      scaleStartValue: 0,
      responsive: true,
      offsetgridlines: true,
      datasetFill: false
    });
  }
}

function populateTable(data) {
  var table = document.getElementById('data-table');
  var rows = table.getElementsByTagName('tr');
  for (var i = rows.length - 1; i > 0; i--) {
    table.removeChild(rows[i]);
  }

  var modelData = data.model_data;
  modelData.forEach((row, index) => {
    var tableRow = document.createElement('tr');
    var numberCell = document.createElement('td');
    numberCell.textContent = index + 1;
    tableRow.appendChild(numberCell);

    var nameCell = document.createElement('td');
    nameCell.textContent = row[0];
    tableRow.appendChild(nameCell);

    for (var i = 1; i < row.length; i++) {
      var td = document.createElement('td');
      var roundedValue = parseFloat(row[i]).toFixed(3);
      td.textContent = roundedValue;
      tableRow.appendChild(td);
    }

    table.appendChild(tableRow);
  });
}

fetchData()
  .then(data => {
    populateTable(data);
    updateChartWithData(data.model_data, selectedMetricValue);
  });
