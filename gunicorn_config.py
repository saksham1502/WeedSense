import multiprocessing

# Timeout settings
timeout = 120  # 2 minutes for model loading
graceful_timeout = 30

# Worker settings
workers = 1  # Free tier has limited RAM
worker_class = "sync"

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Preload app to load model before forking workers
preload_app = True
