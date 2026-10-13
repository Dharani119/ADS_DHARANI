const toggle = document.getElementById("menuToggle");
const nav = document.getElementById("nav");
if (toggle && nav) toggle.addEventListener("click", () => nav.classList.toggle("open"));

const palette = ["#2563eb","#60a5fa","#93c5fd","#1d4ed8","#bfdbfe","#3b82f6","#dbeafe","#1e40af"];
const commonOptions = {
  responsive: true, maintainAspectRatio: false,
  plugins: { legend: { display: false } },
  scales: {
    x: { grid: { display: false }, ticks: { color: "#718096" } },
    y: { beginAtZero: true, grid: { color: "#eaf0f8" }, ticks: { color: "#718096", precision: 0 } }
  }
};
const categoryCanvas = document.getElementById("categoryChart");
if (categoryCanvas && window.pgCategories) {
  new Chart(categoryCanvas, {type:"bar", data:{labels:window.pgCategories.map(x=>x[0]),datasets:[{data:window.pgCategories.map(x=>x[1]),backgroundColor:palette,borderRadius:7,maxBarThickness:32}]},options:commonOptions});
}
const ratingCanvas = document.getElementById("ratingChart");
if (ratingCanvas && window.pgRatings) {
  const labels = Object.keys(window.pgRatings);
  new Chart(ratingCanvas, {type:"bar", data:{labels,datasets:[{data:labels.map(k=>window.pgRatings[k]),backgroundColor:"#3b82f6",borderRadius:7,maxBarThickness:42}]},options:commonOptions});
}
