var colors = [
  "",
  "rgba(233, 79, 55, 1.0)",
  "rgba(255, 252, 49, 1.0)",
  "rgba(109, 163, 77, 1.0)"
];

var selectedMetricValue = "P";

function selectMetric(element, metric) {
  var menuItems = document.querySelectorAll('.menu li');
  for (let i = 1; i < menuItems.length; i++) {
    menuItems[i].classList.remove('selected');
  }
  element.classList.add('selected');
  selectedMetricValue = metric;

  let menustaticList = document.querySelector('.menustatic');
  let listItems = menustaticList.getElementsByTagName('li');
  for (let i = listItems.length - 1; i > 0; i--) {
    menustaticList.removeChild(listItems[i]);
  }
  if(metric!="AUC-ROC"){
    var subscripts = [2,5,10]

    for(let i of subscripts) {
      let listItem = document.createElement('li');
      listItem.innerHTML = `${selectedMetricValue}<sub>${i}</sub>`;
      let colorIcon = document.createElement('span');
      colorIcon.className = 'colorIcon';
      colorIcon.style.backgroundColor = colors[subscripts.indexOf(i)+1];
      listItem.appendChild(colorIcon);
      menustaticList.appendChild(listItem);
    }
  }
  else{
    let listItem = document.createElement('li');
    listItem.innerHTML = `${selectedMetricValue}`;
    let colorIcon = document.createElement('span');
    colorIcon.className = 'colorIcon';
    colorIcon.style.backgroundColor = colors[1];
    listItem.appendChild(colorIcon);
    menustaticList.appendChild(listItem);
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

  var checkedData = [];
  var checkedListItems = [];
  var listItems = document.querySelectorAll("#model-list li");
  listItems.forEach((listItem, index) => {
    const checkbox = listItem.querySelector('input[type="checkbox"]');
    if (checkbox && checkbox.checked) {
      var data = modelData[parseInt(listItem.dataset.position)];
      if (data && data.length > 1) { 
        checkbox.dataset.index = index;
        checkedData.push(data);
        checkedListItems.push(listItem);
      }
    }
  });

  if (checkedData.length === 0) return;

  checkedData.sort((a, b) => {
    var positionA = checkedListItems.findIndex(item => item.dataset.position == a[0]);
    var positionB = checkedListItems.findIndex(item => item.dataset.position == b[0]);
    return positionA - positionB;
  });

  var barChartData = {
    labels: checkedData.map(row => row[0]),
    datasets: []
  };

  var ctx = document.getElementById("graph").getContext("2d");
  if(window.myBarChart) window.myBarChart.destroy();

  if(r != 12){
    var data_2 = checkedData.map(row => parseFloat(row[r+1]).toFixed(3));
    var data_5 = checkedData.map(row => parseFloat(row[r+2]).toFixed(3));
    var data_10 = checkedData.map(row => parseFloat(row[r+3]).toFixed(3));

    barChartData.datasets = [
      {
        label: metric+"2",
        fillColor: colors[1],
        data: data_2
      },
      {
        label: metric+"5",
        fillColor: colors[2],
        data: data_5
      },
      {
        label: metric+"10",
        fillColor: colors[3],
        data: data_10
      }
    ];
  }
  else{
    barChartData.datasets = [
      {
        label: metric,
        fillColor: colors[1],
        data: checkedData.map(row => parseFloat(row[r+1]).toFixed(3))
      }
    ];
  }

  window.myBarChart = new Chart(ctx).Bar(barChartData, {
    responsive : true,
    scaleBeginAtZero : true,
    scaleShowGridLines : true,
    barShowStroke: true,
    barStrokeWidth: 2,
    barValueSpacing: 5,
    barDatasetSpacing : 1,
    scaleOverride: true,
    scaleSteps: 10,
    scaleStepWidth: r != 12 ? 0.01 : 0.1,
    scaleStartValue: 0,
    multiTooltipTemplate: "<%= datasetLabel %> - <%= value %>",
    scaleLabel: "<%= ' ' + value%>", 
    scaleFontColor: "#d3d3d3", 
    scaleFontSize: 13,
    scaleFontFamily: "Lato"
  });
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

    var metricCells = row.slice(1,14).map(value => {
      var cell = document.createElement('td');
      cell.textContent = parseFloat(value).toFixed(3);
      return cell;
    });

    metricCells.forEach(cell => {
      tableRow.appendChild(cell);
    });

    table.appendChild(tableRow);
  });
}

fetchData()
  .then(data => {
    var modelList = document.getElementById('model-list');
    var listItems = modelList.getElementsByTagName('li');
    for (let i = listItems.length - 1; i > 0; i--) {
      modelList.removeChild(listItems[i]);
    }

    const modelData = data.model_data;
    modelData.forEach((row, index) => {
      var listItem = document.createElement('li');
      listItem.textContent = row[0];
      listItem.setAttribute('draggable', true);
      listItem.setAttribute('data-position', index);

      var checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.checked = true;
      checkbox.onchange = function() {
        fetchData()
          .then(data => {
            populateTable(data);
            updateChartWithData(data.model_data, selectedMetricValue);
          });
      };

      listItem.prepend(checkbox);
      modelList.appendChild(listItem);
    });

    new Sortable(modelList, {
      animation: 150,
      onUpdate: function (evt) {
        const modelData = JSON.parse(localStorage.getItem('modelData'));
        const { oldIndex, newIndex } = evt;
        const movedItem = modelData.splice(oldIndex, 1)[0];
        modelData.splice(newIndex, 0, movedItem);
        localStorage.setItem('modelData', JSON.stringify(modelData));
        fetchData()
          .then(data => {
            populateTable(data);
            updateChartWithData(data.model_data, selectedMetricValue);
          });
      }
    });

    populateTable(data);
    updateChartWithData(data.model_data, selectedMetricValue);
  });
