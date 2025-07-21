BASEDIR="$(dirname $0)"

# Build the Docker image
docker build -t ardcnn:cuda12 -f "$BASEDIR/Dockerfile" "$BASEDIR"

echo "Docker image ardcnn:cuda12 was built"
echo "Run build-cuda to run the model"