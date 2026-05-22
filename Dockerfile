FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app/src

COPY requirements.txt .

# requirements.txt is UTF-16 LE (Windows pip freeze output).
# Decode to UTF-8 and drop pywin32, which has no Linux wheel.
RUN python -c "\
import codecs, pathlib; \
txt = codecs.open('requirements.txt', encoding='utf-16').read(); \
lines = [l for l in txt.splitlines() if l.strip() and not l.lower().startswith('pywin32')]; \
pathlib.Path('requirements_clean.txt').write_text('\n'.join(lines))" \
 && pip install --no-cache-dir -r requirements_clean.txt \
 && pip install --no-cache-dir fastapi mcp-server-sqlite aiosqlite \
 && rm requirements_clean.txt

COPY src/ ./src/

# Create runtime directories and a non-root user so the process does not run as root.
RUN mkdir -p skills chroma_data published_sites \
    && groupadd -r agent && useradd -r -g agent -s /sbin/nologin agent \
    && chown -R agent:agent /app

USER agent

CMD ["uvicorn", "gateway:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
