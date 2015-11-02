__author__ = 'Andrei'

import sys
import time
import logging
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler, FileSystemEventHandler
from datetime import datetime
import subprocess
from time import sleep


class MyEventHandler(FileSystemEventHandler):

    def on_any_event(self, event):
        if not '~' in event.src_path:
            message = "\"%s: %s %s\"" % (datetime.now(), event.event_type, event.src_path)
            print message
            bash_command = "git commit -m %s" % message
            print bash_command
            subprocess.Popen('git add . --ignore-removal', cwd=path)
            sleep(5)
            subprocess.Popen(bash_command, cwd=path)
            sleep(5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    path = sys.argv[1] if len(sys.argv) > 1 else '.'
    # path = 'C:\\Users\\Andrei\\Desktop\\terrible_git\\Myfolder'
    event_handler = MyEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()