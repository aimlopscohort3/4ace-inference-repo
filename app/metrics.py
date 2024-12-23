from prometheus_client import Summary, Gauge

REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
GENERATION_LENGTH = Gauge('generation_length', 'Length of generated text')