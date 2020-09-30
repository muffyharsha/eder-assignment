docker build -t docker_eder_postgress_image_landing ./landing_db
docker build -t docker_eder_postgress_image_target ./target_db
docker build -t docker_eder_miroservices_image ./microservices

docker run -d -p 5432:5432 docker_eder_postgress_image_landing
docker run -d -p 5434:5432  docker_eder_postgress_image_target
docker run -d -p 5000:5000  docker_eder_miroservices_image
