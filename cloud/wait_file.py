"""Queue guard job: block until a file exists (e.g. the merged 8B E18.1 model) - cloud 10-07."""
import os, sys, time
while not os.path.exists(sys.argv[1]):
    time.sleep(60)
