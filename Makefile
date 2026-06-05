export PYTHONPATH := src

.PHONY: demo test eval api docker

demo:
	python -m finsight.demo

test:
	python -m unittest discover -s tests -p "test_*.py" -v

eval:
	python -m finsight.eval.run_eval --gate 0.8

api:
	uvicorn finsight.api:app --reload

docker:
	docker compose up --build
