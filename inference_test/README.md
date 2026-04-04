1. Run Docker Desktop
2. docker build -f dockerfile.inference -t tbclassifier .
3. docker run -p 8000:8000 --env-file .env tbclassifier
4. cd inference_test
5. python test-inference.py 
