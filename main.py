import uvicorn

uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,        # Auto-reload on code changes (good for dev)
        log_level="info"    # Matches the logger you configured
    )
