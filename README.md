deploy api.py

pip install "uvicorn[standard]"
uvicorn api:app --reload

docker build -t glaucoma-detector .
docker run -p 7860:7860 glaucoma-detector