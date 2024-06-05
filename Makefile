run:
	uvicorn api.api:app --reload
docker_run:
	docker run --env-file .env -p 8000:8000 ${GAR_IMAGE}:dev
docker_build:
	docker build -t  $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/garden-guru/$GAR_IMAGE:prod .
	