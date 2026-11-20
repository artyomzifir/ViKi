FROM viki-sam2

COPY experiments/requirements.txt /tmp/experiment-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/experiment-requirements.txt

WORKDIR /app
ENV PYTHONPATH=/app
