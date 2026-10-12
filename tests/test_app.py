"""Headless smoke test of the operator console (Streamlit AppTest)."""
import os

os.environ["GRIDMIND_LLM"] = "off"
os.environ.setdefault("OMP_NUM_THREADS", "1")

from streamlit.testing.v1 import AppTest


def _btn(at, text):
    return next(b for b in at.sidebar.button if text in b.label)


def test_console_runs_ticks_events_and_documents():
    at = AppTest.from_file("../app/streamlit_app.py", default_timeout=180).run()
    assert not at.exception
    _btn(at, "1 hour").click().run()                       # 4 control cycles
    assert not at.exception
    assert any("Action cluster" in m.value for m in at.markdown)
    _btn(at, "Inject").click().run()                       # default event from the injector
    _btn(at, "1 tick").click().run()
    assert not at.exception
    storm = next(b for b in at.button if "imd_storm_bulletin.pdf" in b.label)
    storm.click().run()                                    # perception on a PDF
    _btn(at, "1 tick").click().run()
    assert not at.exception
    assert any("Storm" in str(m.value) for m in at.metric)
