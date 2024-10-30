from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/classify-email/")
async def say_hello(body: str):
    return {"message": f"{body}"}
