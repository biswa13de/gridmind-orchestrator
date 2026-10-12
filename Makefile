PY ?= python
export OMP_NUM_THREADS=1

.PHONY: install app test eval backtest chaos montecarlo forecast demo data

install:            ## install pinned dependencies + the package
	pip install -r requirements.txt && pip install -e .

app:                ## launch the operator console on http://localhost:8501
	streamlit run app/streamlit_app.py

test:               ## unit + UI smoke tests
	$(PY) -m pytest -q

demo:               ## headless demo storyline -> logs/sample_demo_day.jsonl
	$(PY) scripts/demo_story.py

forecast:           ## held-out forecast calibration
	$(PY) eval/forecast_eval.py

backtest:           ## 30-day held-out backtest, 4 controllers x calm/stress
	$(PY) eval/backtest.py --days 30 --jobs 8

chaos:              ## 12 named fault scenarios
	$(PY) eval/chaos.py

montecarlo:         ## 100 runs x 48 h x 5 strategies
	$(PY) eval/montecarlo.py --runs 100 --hours 48 --jobs 8

eval: forecast backtest chaos montecarlo   ## everything in eval/results/

data:               ## re-download real weather from Open-Meteo
	$(PY) scripts/fetch_weather.py
