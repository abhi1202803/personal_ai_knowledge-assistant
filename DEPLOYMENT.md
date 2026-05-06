# Deployment

This project is ready to deploy as one Docker web service. The backend serves the built React app and the `/api` routes.

## Render

1. Push this repository to GitHub.
2. In Render, create a new Blueprint or Docker Web Service from this repository.
3. Set the private environment variable:
   - `GROQ_API_KEY`: your Groq API key
4. Deploy.

Do not commit `.env` or real API keys to GitHub. Keep keys in the hosting platform's environment variables.

The free Render filesystem is ephemeral. Uploaded knowledge-base documents may disappear after redeploys/restarts unless you attach a persistent disk and set `CHROMADB_PATH` to a path inside that disk.
