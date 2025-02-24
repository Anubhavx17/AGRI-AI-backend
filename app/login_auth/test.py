from app import db
from app.data_models.models import UserModel

user = UserModel(name="Anubhav", email="pawananubhav12@gmail.com")
user.set_password("Anubhav@123")  # This will hash the password
user.confirmed = True

db.session.add(user)
db.session.commit()
