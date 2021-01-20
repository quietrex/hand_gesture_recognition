# Hand Gesture Recognition model on Flask
Deploying hand gesture recognition model into flask app

4 main classes - [Stop, Start, Garbage, Next]

### Download the model and put it into ../model/
[model](https://drive.google.com/file/d/1A8MlFToaRnuSxGRz5SfnKRoH6i2YxG_v/view?usp=sharing)

### To set up virtualenv
```bash
python3 -m venv venv
source venv/bin/activate
```

### To install packages
```bash
pip install -r requirement.txt
```

### To run it
```bash
Linux:
export FLASK_APP=app.py
flask run --host=0.0.0.0
```

```bash
Windows:
set FLASK_APP=app.py
flask run --host=0.0.0.0
```
