# Data: what is real, what is synthetic, and why

| Signal | Source | Real or synthetic | Notes |
|---|---|---|---|
| Solar irradiance (GHI), temperature | [Open-Meteo historical archive](https://open-meteo.com/) (ERA5-based), 2025, hourly | **Real** | Fetched for the actual locations of 5 Indian solar parks: Bhadla (RJ), Charanka (GJ), Pavagada (KA), Kurnool (AP) and Rewa (MP). Interpolated to 15 minutes and shifted to IST. Reproduce with `scripts/fetch_weather.py`. |
| Wind speed at 100 m | Open-Meteo, 2025, hourly | **Real** | Muppandal (TN), Kutch (GJ) and Satara (MH). |
| Solar → MW | PV model in `gridmind/data.py` | Derived | Performance ratio 0.82, with a temperature derate of −0.4%/°C. The resulting capacity factor is 16.5%. |
| Wind → MW | Cubic power curve (cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s) | Derived | The resulting capacity factor is 19.3%. |
| Industrial demand | Shift-pattern profiles × weekly × seasonal factors, plus AR(1) noise | **Synthetic** | Four loads: steel (three-shift), cement (two-shift), data centre (flat) and textile (day shift). Seeded and deterministic. |
| Market price (₹/MWh) | IEX-like day-ahead shape | **Synthetic** | The shape has a solar-hour trough, a morning shoulder and an evening peak, a summer premium, inverse correlation with solar output, AR(1) noise, and rare spikes to the IEX cap of ₹10,000/MWh. The hourly mean runs from ₹3,700 (13:00) to ₹8,500 (20:00), which matches the qualitative IEX pattern. Real IEX 15-minute block data has no free API, so we did not scrape it. |
| Grid carbon intensity (tCO₂/MWh) | Formula anchored to the CEA all-India average of about 0.71–0.75 | **Synthetic** (assumption) | Intensity is lower when solar output is high. |
| Grid frequency | 50 Hz plus AR noise, dipping at scarcity | **Synthetic** | Used for the `grid_stress` situation and the frequency-reserve headroom. |
| Asset specifications | `config/portfolio.yaml` | Assumptions | Plausible values (215 MW solar, 125 MW wind, 130 MWh / 65 MW storage, a 150 MW import interconnection). These are not a real utility's data. |
| Events and documents | `gridmind/scenarios.py`, `data/samples/` | **Synthetic** | Storm bulletins, grid notices and work orders are written in the style of IMD and Grid-India notices. Every event is seeded and reproducible. |

**Train/test discipline.**

- The forecaster is trained on Jan–Aug 2025.
- Its bands are calibrated on Sep 2025.
- All evaluation (backtest, chaos and Monte Carlo) uses Oct–Dec 2025, which the models never saw.

**Third-party software.** NumPy, pandas, SciPy, HiGHS, scikit-learn, Pydantic, Streamlit, Plotly, pypdf, the Anthropic Python SDK and reportlab (used only to generate sample PDFs). See `requirements.txt` for exact versions. Weather data is provided by Open-Meteo under CC BY 4.0.
