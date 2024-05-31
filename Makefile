run:
	uvicorn api.api:app --reload
docker_run:
	docker run --env-file .env -p 8000:8000 ${GAR_IMAGE}:dev