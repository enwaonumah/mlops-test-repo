# We'll use a slim python image as a base
FROM python:3.8.2-slim-buster
 
# Our code will need requests, so we can install them on the image with pip
RUN pip install requests
 
# Add our file and run it
ADD runExecution.py /runExecution.py
CMD ["/runExecution.py"]
ENTRYPOINT ["python"]