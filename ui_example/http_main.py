import uvicorn


def main():
    uvicorn.run(
        "claw.entrypoints.http_app:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
