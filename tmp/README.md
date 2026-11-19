# Temporary workspace

This folder is tracked so the team can see its purpose. Its README is included
in Git. Temporary downloads, logs, local settings and generated files remain ignored.

During development, `tmp/hey-bis-test` was a separate local checkout of the `test`
branch. A normal clone of `test` places the application directly at the repository
root. Use these shared files to run and review the recording feature:

- [Dashboard section](../service/static/index.html)
- [Microphone recording and save controls](../service/static/dataset.js)
- [Server, dataset discovery and manifest registration](../service/server.py)
- [Dataset folder layout and training instructions](../README.md#recording-training-examples)

From the repository root, start the dashboard with:

```bash
python service/server.py --host localhost --port 8080
```

Open <http://localhost:8080/>. Recordings are saved to the nearby
`Dataset/Hey-BIS/audio/recorded/` folder and registered in its `manifest.csv`.
The main README documents the expected folder layout and custom paths.
