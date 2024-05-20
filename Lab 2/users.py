from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class UserRequest(BaseModel):
    username: str
    password: str

@app.post("/register")
def register_user(request: UserRequest):
    # TO DO: implement user registration logic
    return {"message": "User registered successfully"}
