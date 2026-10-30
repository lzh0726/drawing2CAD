import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path):
        self.path = path
        self.lock = threading.RLock()
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, body TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, body TEXT NOT NULL)')
        for job in self.list():
            if job['status'] in ('queued', 'preparing', 'generating', 'checking'):
                self.patch(job['id'], status='interrupted')
                self.event(job['id'], 'system', 'warning', '服务已重启，上一次未完成的任务已中断，可重新运行。')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, job):
        with self.lock, self.connect() as db:
            db.execute('INSERT INTO jobs VALUES (?,?)', (job['id'], json.dumps(job, ensure_ascii=False)))
        return job

    def get(self, job_id):
        with self.connect() as db:
            row = db.execute('SELECT body FROM jobs WHERE id=?', (job_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def list(self):
        with self.connect() as db:
            rows = db.execute('SELECT body FROM jobs ORDER BY rowid DESC').fetchall()
        return [json.loads(r[0]) for r in rows]

    def mutate(self, job_id, fn):
        with self.lock, self.connect() as db:
            row = db.execute('SELECT body FROM jobs WHERE id=?', (job_id,)).fetchone()
            if not row:
                raise KeyError(job_id)
            job = json.loads(row[0])
            fn(job)
            job['updated_at'] = now()
            db.execute('UPDATE jobs SET body=? WHERE id=?', (json.dumps(job, ensure_ascii=False), job_id))
        return job

    def patch(self, job_id, **values):
        return self.mutate(job_id, lambda job: job.update(values))

    def message(self, job_id, role, content):
        return self.mutate(job_id, lambda job: job['messages'].append({'role': role, 'content': content, 'time': now()}))

    def event(self, job_id, stage, level, message):
        event = {'time': now(), 'stage': stage, 'level': level, 'message': message}
        with self.lock, self.connect() as db:
            cursor = db.execute('INSERT INTO events(job_id,body) VALUES (?,?)', (job_id, json.dumps(event, ensure_ascii=False)))
            event['id'] = cursor.lastrowid
        return event

    def events(self, job_id, after=0):
        with self.connect() as db:
            rows = db.execute('SELECT id,body FROM events WHERE job_id=? AND id>? ORDER BY id', (job_id, after)).fetchall()
        return [dict(json.loads(body), id=i) for i, body in rows]
