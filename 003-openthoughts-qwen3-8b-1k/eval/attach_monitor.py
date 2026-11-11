"""Reattach monitoring without restarting the existing evaluation or inference."""
import fcntl
import time
import pipeline as p
from daytona_controller import WORKER_JOB


def main():
    with (p.NODE.parent / '.daytona-evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        attached = False
        while True:
            try:
                current = p.pods(WORKER_JOB)
                p.write(p.RUN / 'monitor-health.json', {'cluster_ok': True, 'updated_at': p.stamp()})
            except Exception as exc:
                # Authentication/control-plane failure must never stop live jobs.
                p.write(p.RUN / 'monitor-health.json', {'cluster_ok': False, 'error': str(exc), 'updated_at': p.stamp()})
                time.sleep(30)
                continue
            if current and any(x['status']['phase'] in ('Running', 'Pending') for x in current):
                if not attached:
                    p.state('running', 'evaluation', 'Reattached to existing evaluation worker; tasks were not restarted')
                    attached = True
                time.sleep(30)
                continue
            if current and all(x['status']['phase'] in ('Succeeded', 'Failed') for x in current):
                # The worker owns final results and sandbox cleanup. Archive only
                # once its pod is terminal; retry cluster failures in place.
                try:
                    p.archive_and_delete(WORKER_JOB, p.RUN / 'worker-logs')
                    p.archive_and_delete(p.SERVE_JOB, p.RUN / 'serving-logs')
                    return
                except Exception as exc:
                    p.write(p.RUN / 'monitor-health.json', {'cluster_ok': False, 'error': str(exc), 'updated_at': p.stamp()})
            time.sleep(30)


if __name__ == '__main__':
    main()
