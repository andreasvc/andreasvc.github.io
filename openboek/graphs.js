function alertFunction() {
  alert("This will be made available soon!");
}


Chart.defaults.font.size = 18;
var ctx = document.getElementById("myChart");
var myChart = new Chart(ctx, {
    type: 'bar',
    data: {
        labels: ['BookX', 'BookY', 'BookZ', 'BookF', 'BookH', 'BookL'],
        datasets: [{
            label: '# of ?',
            data: [12, 19, 3, 5, 2, 3],
            backgroundColor: [
                'rgba(255, 99, 132, 0.2)',
                'rgba(54, 162, 235, 0.2)',
                'rgba(255, 206, 86, 0.2)',
                'rgba(75, 192, 192, 0.2)',
                'rgba(153, 102, 255, 0.2)',
                'rgba(255, 159, 64, 0.2)'
            ],
            borderColor: [
                'rgba(255, 99, 132, 1)',
                'rgba(54, 162, 235, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(153, 102, 255, 1)',
                'rgba(255, 159, 64, 1)'
            ],
            borderWidth: 1
        }]
    },
    options: {
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});

var ctx = document.getElementById("radar-chart");
var myChart = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['Anna Karenina', 'De Ellendigen', ['De reis om de wereld','in tachtig dagen'], 'De uitvreter', 'Dichtertje', 'Eline Vere', 'Max Havelaar', ['Sherlock Holmes en', 'de Agra-schat'], 'Titaantjes'],
      datasets: [
        {
          label: "Mentions",
          fill: true,
          backgroundColor: "rgba(179,181,198,0.2)",
          borderColor: "rgba(179,181,198,1)",
          pointBorderColor: "#fff",
          pointBackgroundColor: "rgba(179,181,198,1)",
          data: [2323,2414,2218,3136,4118,2272,2231,2388,2534]
        }, {
          label: "Entities",
          fill: true,
          backgroundColor: "rgba(255,99,132,0.2)",
          borderColor: "rgba(255,99,132,1)",
          pointBorderColor: "#fff",
          pointBackgroundColor: "rgba(255,99,132,1)",
          pointBorderColor: "#fff",
          data: [735,1063,932,1266,1550,875,800,893,811]
        }
      ]
    },
    options: {
    legend: {
      labels: {
        fontColor: '#11c091'
      }
    },
    scale: {
      gridLines: {
        color: '#AAA'
      },
      ticks: {
        beginAtZero: false,
        max: 4200,
        min: 0,
        stepSize: 1000,
        fontColor: '#fff',
        backdropColor: '#444'
      },
      pointLabels: {
        fontSize: 12,
        fontColor: '#11c091'
      }
    }
  }
});