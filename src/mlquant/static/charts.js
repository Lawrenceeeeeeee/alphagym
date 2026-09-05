/* Render ECharts figures from data embedded in DOM elements.
 *
 * Two element kinds are supported:
 *  - .echarts-nav[data-series]: layered NAV line chart per factor variant,
 *    data-series is a JSON string like {"series": [{name, color, dates, values}]}.
 *  - .echarts-report[data-kind]: standalone research report charts
 *    (correlation heatmap / combination NAV), see report.html.
 */
(function () {
  "use strict";

  function drawNav(element) {
    var payload = JSON.parse(element.getAttribute("data-series"));
    var chart = echarts.init(element, null, { renderer: "canvas" });
    chart.setOption({
      title: {
        text: element.getAttribute("data-title") || "五分组累计净值",
        left: "left",
        textStyle: { fontSize: 13, color: "#1c2320" },
      },
      tooltip: { trigger: "axis" },
      legend: {
        top: 28, left: "left", type: "scroll",
        textStyle: { fontSize: 11, color: "#3c4640" },
        itemWidth: 14, itemHeight: 8,
      },
      grid: { left: 44, right: 16, top: 64, bottom: 40 },
      xAxis: {
        type: "category",
        data: payload.series.length ? payload.series[0].dates : [],
        axisLabel: { fontSize: 10, color: "#5b6560" },
        axisLine: { lineStyle: { color: "#d9ddd7" } },
      },
      yAxis: {
        type: "value", scale: true, name: "累计净值（倍）",
        nameTextStyle: { fontSize: 10, color: "#5b6560" },
        axisLabel: { fontSize: 10, color: "#5b6560" },
        splitLine: { lineStyle: { color: "#e4e8e2" } },
      },
      series: payload.series.map(function (item) {
        return {
          name: item.name,
          type: "line",
          data: item.values,
          symbol: "none",
          lineStyle: { width: item.line_width || 1.5, color: item.color },
          itemStyle: { color: item.color },
          emphasis: { disabled: true },
          markLine: {
            silent: true, symbol: "none",
            lineStyle: { color: "#aeb7b2", type: "dashed", width: 0.8 },
            data: [{ yAxis: 1 }],
          },
        };
      }),
    });
    chart.resize();
  }

  function drawReport(element) {
    var kind = element.getAttribute("data-kind");
    var payload = JSON.parse(element.getAttribute("data-series"));
    var chart = echarts.init(element, null, { renderer: "canvas" });
    if (kind === "correlation") {
      var labels = payload.labels;
      var cells = payload.cells.map(function (cell) {
        return [cell[1], cell[0], cell[2]];
      });
      chart.setOption({
        title: {
          text: "因子 Spearman 秩相关（月度截面平均）",
          left: "left", textStyle: { fontSize: 13, color: "#1c2320" },
        },
        tooltip: {
          position: "top",
          formatter: function (params) {
            return labels[params.value[1]] + " × " + labels[params.value[0]]
              + "：<b>" + params.value[2].toFixed(3) + "</b>";
          },
        },
        grid: { left: 90, right: 20, top: 30, bottom: 110 },
        xAxis: {
          type: "category", data: labels, splitArea: { show: true },
          axisLabel: { fontSize: 9, color: "#3c4640", rotate: 60, interval: 0 },
          axisLine: { show: false }, axisTick: { show: false },
        },
        yAxis: {
          type: "category", data: labels, splitArea: { show: true },
          inverse: true,
          axisLabel: { fontSize: 9, color: "#3c4640" },
          axisLine: { show: false }, axisTick: { show: false },
        },
        visualMap: {
          min: -1, max: 1, calculable: false, orient: "horizontal",
          left: "center", bottom: 10,
          inRange: { color: ["#2c7a4b", "#fffef9", "#b42318"] },
          textStyle: { fontSize: 10, color: "#5b6560" },
        },
        series: [{
          name: "Spearman",
          type: "heatmap",
          data: cells,
          label: {
            show: labels.length <= 8, fontSize: 8,
            formatter: function (params) {
              return params.value[2].toFixed(2);
            },
          },
          emphasis: { itemStyle: { borderColor: "#26322d", borderWidth: 1 } },
        }],
      });
    } else if (kind === "combination") {
      chart.setOption({
        title: {
          text: "不同合成方式多空累计净值（权重在验证期末冻结，测试期仅评估）",
          left: "left", textStyle: { fontSize: 13, color: "#1c2320" },
        },
        tooltip: { trigger: "axis" },
        legend: {
          top: 28, left: "left", type: "scroll",
          textStyle: { fontSize: 11, color: "#3c4640" },
        },
        grid: { left: 44, right: 16, top: 64, bottom: 40 },
        xAxis: {
          type: "category",
          data: payload.dates,
          axisLabel: { fontSize: 10, color: "#5b6560" },
          axisLine: { lineStyle: { color: "#d9ddd7" } },
        },
        yAxis: {
          type: "value", scale: true, name: "累计净值（倍）",
          nameTextStyle: { fontSize: 10, color: "#5b6560" },
          axisLabel: { fontSize: 10, color: "#5b6560" },
          splitLine: { lineStyle: { color: "#e4e8e2" } },
        },
        series: payload.series.map(function (item) {
          return {
            name: item.name, type: "line", data: item.values,
            symbol: "none", lineStyle: { width: 1.8, color: item.color },
            itemStyle: { color: item.color },
            markLine: {
              silent: true, symbol: "none",
              lineStyle: { color: "#aeb7b2", type: "dashed", width: 0.8 },
              data: [{ yAxis: 1 }],
            },
          };
        }),
      });
    }
    chart.resize();
  }

  function init() {
    document.querySelectorAll(".echarts-nav").forEach(drawNav);
    document.querySelectorAll(".echarts-report").forEach(drawReport);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
