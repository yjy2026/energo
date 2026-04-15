/* energo Static Dashboard — JavaScript */

const DATA_PATH = './data';
let forecastChart = null;
let forecastData = {};

document.addEventListener('DOMContentLoaded', async () => {
    await loadAllData();
    renderMarketStatus();
    renderForecast(24);
    renderScheduleResult();
    renderStrategyComparison();
    setupControls();
});

// ── Data Loading ─────────────────────────────────────────────
async function loadAllData() {
    const files = ['market', 'forecast_12', 'forecast_24', 'forecast_48', 'schedule_example', 'compare'];
    const results = await Promise.all(
        files.map(f => fetch(`${DATA_PATH}/${f}.json`).then(r => r.json()))
    );
    files.forEach((f, i) => { forecastData[f] = results[i]; });
}

// ── Market Status ────────────────────────────────────────────
function renderMarketStatus() {
    const d = forecastData.market;
    if (!d || d.error) return;

    document.getElementById('kpi-price').textContent = d.latest_price.toFixed(1);
    document.getElementById('kpi-avg').textContent = d.today.avg.toFixed(1);
    document.getElementById('kpi-vol').textContent = d.week.volatility.toFixed(1);

    const el = document.getElementById('kpi-trend');
    const rising = d.week.trend === 'rising';
    el.textContent = rising ? '↑ Rising' : '↓ Falling';
    el.style.color = rising ? '#ff6b6b' : '#00d4aa';
}

// ── Forecast Chart ───────────────────────────────────────────
function renderForecast(hours) {
    const key = `forecast_${hours}`;
    const data = forecastData[key];
    if (!data || data.error) return;

    const labels = data.slots.map(s => `+${s.hour_offset}h`);
    const means = data.slots.map(s => s.price_mean);
    const uppers = data.slots.map(s => s.ci_90_upper);
    const lowers = data.slots.map(s => s.ci_90_lower);

    if (forecastChart) forecastChart.destroy();

    const ctx = document.getElementById('forecast-chart').getContext('2d');
    forecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: '90% CI Upper',
                    data: uppers,
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(0, 150, 255, 0.08)',
                    fill: '+1',
                    pointRadius: 0,
                    tension: 0.3,
                },
                {
                    label: 'Predicted Price (μ)',
                    data: means,
                    borderColor: '#00d4aa',
                    backgroundColor: 'rgba(0, 212, 170, 0.05)',
                    borderWidth: 2.5,
                    fill: false,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    pointHoverBackgroundColor: '#00d4aa',
                    tension: 0.3,
                },
                {
                    label: '90% CI Lower',
                    data: lowers,
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(0, 150, 255, 0.08)',
                    fill: '-1',
                    pointRadius: 0,
                    tension: 0.3,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(16, 22, 36, 0.95)',
                    titleColor: '#e8ecf4',
                    bodyColor: '#8892a6',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        label: (ctx) => {
                            if (ctx.datasetIndex === 1) {
                                return `Price: ¥${ctx.parsed.y.toFixed(2)}/kWh`;
                            }
                            return null;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.03)' },
                    ticks: { color: '#5a6478', font: { size: 11 }, maxTicksLimit: 12 },
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.03)' },
                    ticks: { color: '#5a6478', font: { size: 11 }, callback: v => `¥${v}` },
                },
            },
        },
    });
}

// ── Schedule Result ──────────────────────────────────────────
function renderScheduleResult() {
    const d = forecastData.schedule_example;
    if (!d || d.error) return;

    const wl = d.workload;
    const cost = d.cost_analysis;
    const sched = d.optimal_schedule;

    document.getElementById('wl-name').textContent = wl.name;
    document.getElementById('wl-duration').textContent = `${wl.duration_hours}h`;
    document.getElementById('wl-power').textContent = `${wl.power_kw} kW`;
    document.getElementById('wl-energy').textContent = `${wl.energy_kwh} kWh`;

    document.getElementById('result-start').textContent = `${sched.start_hours_from_now}h`;
    document.getElementById('result-baseline').textContent = `¥${cost.baseline_cost_jpy.toLocaleString()}`;
    document.getElementById('result-cost').textContent = `¥${cost.predicted_cost_jpy.toLocaleString()}`;
    document.getElementById('result-savings').textContent = `¥${cost.savings_jpy.toLocaleString()} (${cost.savings_pct}%)`;
    document.getElementById('result-cvar').textContent = `¥${cost.cvar_cost_jpy.toLocaleString()}`;
    document.getElementById('result-risk-adj').textContent = `¥${cost.risk_adjusted_cost_jpy.toLocaleString()}`;
    document.getElementById('result-recommendation').textContent = d.recommendation;
}

// ── Strategy Comparison ──────────────────────────────────────
function renderStrategyComparison() {
    const d = forecastData.compare;
    if (!d || d.error) return;

    const container = document.getElementById('strategy-table');
    let html = `<div class="strategy-row header">
        <span>Strategy</span><span>Start</span><span>Cost</span><span>Savings</span>
    </div>`;

    const labels = {
        aggressive: '🔥 Aggressive',
        balanced: '⚖️ Balanced',
        conservative: '🛡️ Conservative',
        max_safety: '🔒 Max Safety',
    };

    for (const [key, val] of Object.entries(d.strategies)) {
        html += `<div class="strategy-row">
            <span class="strategy-name">${labels[key] || key} <span class="alpha">α=${val.alpha}</span></span>
            <span>+${val.start_hours_from_now}h</span>
            <span>¥${val.predicted_cost_jpy.toLocaleString()}</span>
            <span style="color:#00d4aa">${val.savings_pct}%</span>
        </div>`;
    }

    container.innerHTML = html;
}

// ── Controls ─────────────────────────────────────────────────
function setupControls() {
    document.querySelectorAll('[data-hours]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('[data-hours]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            renderForecast(parseInt(btn.dataset.hours));
        });
    });
}
