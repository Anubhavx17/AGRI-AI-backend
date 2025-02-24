from flask import Blueprint, request, jsonify, url_for, current_app, redirect
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from app import db
from app.data_models.models import UserModel
import jwt as pyjwt
from datetime import timedelta
from .email import send_registration_link,send_reset_password_link

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')

        # Check if the email already exists in the database
        existing_user = UserModel.query.filter_by(email=email).first()

        if existing_user:
            if not existing_user.confirmed:
                return jsonify({"msg": "Email not confirmed!"}), 201  # Email exists but not confirmed

            return jsonify({"msg": "Email already registered"}), 201  # Email exists and confirmed

        # Create a new user instance
        user = UserModel(name=name, email=email)
        user.set_password(password)

        # Add the user to the database
        db.session.add(user)
        db.session.commit()

        # Create a token for email confirmation
        token = create_access_token(identity=user.id, expires_delta=timedelta(hours=1))

        # Send registration confirmation email
        if send_registration_link(email, token, name):
            return jsonify({"success": True}), 201
        else:
            # If the email fails to send, delete the user from the database
            db.session.delete(user)
            db.session.commit()
            return jsonify({"success": False, "msg": "Failed to send confirmation email"}), 201
    except Exception as e:
        # Log the exception for debugging purposes
        print(f"An error occurred during registration: {e}")
        return jsonify({"success": False, "msg": "An unexpected error occurred"}), 201
    

@auth_bp.route('/confirm/<token>', methods=['GET'])
def confirm_email(token):
    try:
        # Decode the token to get the user ID
        user_id = pyjwt.decode(token, current_app.config['JWT_SECRET_KEY'], algorithms=["HS256"])['sub']
        
        # Find the user by ID
        user = UserModel.query.get(user_id)
        
        # Check if the user exists
        if not user:
            return jsonify({"msg": "Invalid or expired confirmation link"}), 400
        
        # Check if the account is already confirmed
        if user.confirmed:
            return jsonify({"msg": "Account already confirmed"}), 400
        
        # Confirm the user's account
        user.confirmed = True
        db.session.commit()
        
        # Redirect to the front-end confirmation page
        return redirect('http://localhost:3000/EmailVerified')

    except pyjwt.ExpiredSignatureError:
        return jsonify({"msg": "Confirmation link expired"}), 400
    except pyjwt.InvalidTokenError:
        return jsonify({"msg": "Invalid confirmation link"}), 400
    except Exception as e:
        # Log the exception for debugging purposes
        print(f"An error occurred during email confirmation: {e}")
        return jsonify({"msg": "An unexpected error occurred"}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    print(data)
    email = data.get('email')
    password = data.get('password')

    # Check if the user exists
    user = UserModel.query.filter_by(email=email).first()

    if user is None:
        return jsonify({"msg": "Email does not exist!"}), 201  # Unauthorized status code

    # Check if the user's email is confirmed
    if not user.confirmed:
        return jsonify({"msg": "Email not confirmed!"}), 201  # Unauthorized status code
    
    # Check if the password is correct
    if not user.check_password(password):
        return jsonify({"msg": "Invalid Credentials!"}), 201  # Unauthorized status code

    # If everything is fine, create the access token
    access_token = create_access_token(identity=user.id)
    return jsonify(access_token=access_token), 200


@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    return jsonify({"msg": "Logged out"}), 200

@auth_bp.route('/forgot_password', methods=['POST'])
def forgot_password():
    try:
        data = request.get_json()
        email = data.get('email')

        # Check if the user exists
        user = UserModel.query.filter_by(email=email).first()

        if user is None:
            return jsonify({"msg": "Email does not exist!"}), 201  # Unauthorized status code

        # Generate a token for password reset, valid for 1 hour
        token = create_access_token(identity=user.id, expires_delta=timedelta(hours=1))

        # Generate the reset password URL pointing to the frontend
        frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
        reset_url = f"{frontend_url}/reset_password/{token}"

        # Send password reset email
        if send_reset_password_link(email, reset_url, user.name):
            return jsonify({"success": True, "msg": "Password reset email sent"}), 200
        else:
            return jsonify({"success": False, "msg": "Failed to send reset email"}), 201
    except Exception as e:
        # Log the exception for debugging purposes
        print(f"An error occurred during password reset request: {e}")
        return jsonify({"success": False, "msg": "An unexpected error occurred, Please try again"}), 201


@auth_bp.route('/reset_password/<token>', methods=['POST'])
def reset_password(token):
    try:
        data = request.get_json()
        new_password = data.get('new_password')

        # Decode the token to get the user ID
        user_id = pyjwt.decode(token, current_app.config['JWT_SECRET_KEY'], algorithms=["HS256"])['sub']

        # Find the user by ID
        user = UserModel.query.get(user_id)

        if not user:
            return jsonify({"msg": "Invalid or expired reset link"}), 201

        # Set the new password
        user.set_password(new_password)
        db.session.commit()

        return jsonify({"success": True, "msg": "Password reset successful"}), 200

    except pyjwt.ExpiredSignatureError:
        return jsonify({"msg": "Reset link expired"}), 200
    except pyjwt.InvalidTokenError:
        return jsonify({"msg": "Invalid reset link"}), 200
    except Exception as e:
        # Log the exception for debugging purposes
        print(f"An error occurred during password reset: {e}")
        return jsonify({"msg": "An unexpected error occurred"}), 201
    
@auth_bp.route('/user_details', methods=['GET'])
@jwt_required()
def get_projects():
    user_id = get_jwt_identity()
    user = UserModel.query.get(user_id)
    return jsonify({"name": user.name,"email": user.email}), 200