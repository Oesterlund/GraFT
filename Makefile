test:
	pip install . && pytest -s -v

run-line-profiler:
	pip install . && kernprof -l -v src/graft/cli.py timeseries src/graft/tiff/timeseries.tif /tmp/graft_output
	rm cli.py.lprof && rm -rf /tmp/graft_output
