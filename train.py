#!/usr/bin/env python

import subprocess

for i in range(10000):
    with subprocess.Popen(['python', 'train_epoch.py', str(i)]) as p:
        try:
            code = p.wait()
            if code != 0:
                break
        except KeyboardInterrupt:
            p.send_signal(subprocess.signal.SIGINT)
            p.wait()
            break

print('done')