test:
	pip install . && pytest -s -v

run-webapp:
	pip install . && graft-webapp

run-line-profiler:
	pip install . && kernprof -l -v src/graft/cli.py timeseries src/graft/tiff/timeseries.tif /tmp/graft_output
	rm cli.py.lprof && rm -rf /tmp/graft_output

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf src/*.egg-info/ tests/__pycache__/
