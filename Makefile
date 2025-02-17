install:
	pip install .

test:
	pip install . && pytest -s -v

run-webapp:
	pip install . && graft-webapp

run-cli-example:
	time python src/graft/cli.py timeseries src/graft/data/timeseries.tif /tmp/graft_output
	rm -rf /tmp/graft_output

create-cpu-profiles: install
	python -m cProfile -o timeseries.cprof src/graft/cli.py timeseries --disable_parallelization src/graft/data/timeseries.tif /tmp/graft_output && rm -rf /tmp/graft_output
	python -m cProfile -o timeseries-parallel.cprof src/graft/cli.py timeseries src/graft/data/timeseries.tif /tmp/graft_output && rm -rf /tmp/graft_output

run-line-profiler:
	pip install . && kernprof -l -v src/graft/cli.py timeseries src/graft/data/timeseries.tif /tmp/graft_output
	rm cli.py.lprof && rm -rf /tmp/graft_output

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf src/*.egg-info/ tests/__pycache__/
