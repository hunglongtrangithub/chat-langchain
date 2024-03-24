from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


if __name__ == "__main__":
    import os
    import uvicorn

    port = os.getenv("PORT", 8080)
    uvicorn.run(app, host="0.0.0.0", port=int(port))
