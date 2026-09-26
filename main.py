
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import joblib
import string

app = FastAPI()

templates = Jinja2Templates(directory="templates")

model = joblib.load("password_model.pkl")


def extract_features(password):
    return [[
        len(password),
        sum(c.isupper() for c in password),
        sum(c.islower() for c in password),
        sum(c.isdigit() for c in password),
        sum(c in string.punctuation for c in password)
    ]]


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"strength": None}
    )


@app.post("/predict")
def predict(request: Request, password: str = Form(...)):
    features = extract_features(password)
    prediction = model.predict(features)[0]

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"strength": prediction}
    )