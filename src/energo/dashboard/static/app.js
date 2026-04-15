/* energo Dashboard — Main JavaScript */

// ── State ────────────────────────────────────────────────────
let forecastChart = null;
let currentHorizon = 24; // Must match the 'active' button in HTML

// ── Init ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadMarketStatus();
    loadForecast(currentHorizon);
    setupEventListeners();
    setInterval(loadMarketStatus, 60000);
    setInterval(() => loadForecast(currentHorizon), 60000);
});

// ── Event Listeners ──────────────────────────────────────────
function setupEventListeners() {
    // Chart horizon buttons
    document.querySelectorAll('[data-hours]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('[data-hours]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentHorizon = parseInt(btn.dataset.hours);
            loadForecast(currentHorizon);
        });
    });

    // Risk slider
    const slider = document.getElementById('input-risk');
    const display = document.getElementById('risk-value');
    slider.addEventListener('input', () => {
        display.textContent = parseFloat(slider.value).toFixed(1);
    });

    // Schedule form
    document.getElementById('schedule-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        await runScheduler();
    });

    // Refresh button
    document.getElementById('btn-refresh-data').addEventListener('click', async () => {
        const btn = document.getElementById('btn-refresh-data');
        btn.textContent = '⏳ Refreshing...';
        btn.disabled = true;
        try {
            await fetch('/api/data/refresh', { method: 'POST' });
            await loadMarketStatus();
            await loadForecast(currentHorizon);
        } finally {
            btn.textContent = '↻ Refresh';
            btn.disabled = false;
        }
    });
}

// ── Market Status ────────────────────────────────────────────
async function loadMarketStatus() {
    try {
        const res = await fetch('/api/market/status');
        const data = await res.json();

        if (data.error) return;

        document.getElementById('kpi-price').textContent = data.latest_price.toFixed(1);
        document.getElementById('kpi-avg').textContent = data.today.avg.toFixed(1);
        document.getElementById('kpi-vol').textContent = data.week.volatility.toFixed(1);

        const trendEl = document.getElementById('kpi-trend');
        const isRising = data.week.trend === 'rising';
        trendEl.textContent = isRising ? '↑ Rising' : '↓ Falling';
        trendEl.style.color = isRising ? '#ff6b6b' : '#00d4aa';

        // Data status
        const statusRes = await fetch('/api/data/status');
        const statusData = await statusRes.json();
        const statusEl = document.getElementById('data-status');
        if (statusData.minutes_since_refresh !== undefined) {
            const mins = statusData.minutes_since_refresh;
            statusEl.textContent = mins < 60
                ? `Updated ${Math.round(mins)} min ago`
                : `Updated ${Math.round(mins / 60)}h ago`;
        }
    } catch (err) {
        console.error('Failed to load market status:', err);
    }
}

// ── Forecast Chart ───────────────────────────────────────────
async function loadForecast(hours) {
    try {
        const res = await fetch(`/api/forecast/${hours}`);
        const data = await res.json();

        if (data.error) return;

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
                        ticks: {
                            color: '#5a6478',
                            font: { size: 11 },
                            maxTicksLimit: 12,
                        },
                    },
                    y: {
                        grid: { color: 'rgba(255,255,255,0.03)' },
                        ticks: {
                            color: '#5a6478',
                            font: { size: 11 },
                            callback: (v) => `¥${v}`,
                        },
                    },
                },
            },
        });
    } catch (err) {
        console.error('Failed to load forecast:', err);
    }
}

// ── Workload Scheduler ───────────────────────────────────────
async function runScheduler() {
    const btn = document.getElementById('btn-schedule');
    btn.textContent = 'Optimizing...';
    btn.disabled = true;

    try {
        const body = {
            name: document.getElementById('input-name').value,
            duration_hours: parseFloat(document.getElementById('input-duration').value),
            power_kw: parseFloat(document.getElementById('input-power').value),
            risk_aversion: parseFloat(document.getElementById('input-risk').value),
        };

        const res = await fetch('/api/schedule', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        const data = await res.json();
        if (data.error) {
            alert(data.error);
            return;
        }

        const cost = data.cost_analysis;
        const schedule = data.optimal_schedule;

        document.getElementById('result-start').textContent = `${schedule.start_hours_from_now}h`;
        document.getElementById('result-baseline').textContent = `¥${cost.baseline_cost_jpy.toLocaleString()}`;
        document.getElementById('result-cost').textContent = `¥${cost.predicted_cost_jpy.toLocaleString()}`;
        document.getElementById('result-savings').textContent =
            `¥${cost.savings_jpy.toLocaleString()} (${cost.savings_pct}%)`;
        document.getElementById('result-cvar').textContent = `¥${cost.cvar_cost_jpy.toLocaleString()}`;
        document.getElementById('result-risk-adj').textContent = `¥${cost.risk_adjusted_cost_jpy.toLocaleString()}`;
        document.getElementById('result-alpha').textContent = `α=${body.risk_aversion}`;
        document.getElementById('result-recommendation').textContent = data.recommendation;

        document.getElementById('schedule-result').classList.remove('hidden');
    } catch (err) {
        console.error('Schedule error:', err);
        alert('Failed to schedule workload.');
    } finally {
        btn.textContent = 'Optimize Schedule';
        btn.disabled = false;
    }
}
