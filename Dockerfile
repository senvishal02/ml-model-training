FROM public.ecr.aws/lambda/python:3.9

# Install dependencies
RUN pip install --upgrade pip
COPY src/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# copy training code
COPY src /opt/ml/code

ENV PYTHONPATH=/opt/ml/code

# Entry for training (used by CodeBuild or custom)
ENTRYPOINT ["python", "/opt/ml/code/train.py"]
