from flask import current_app, url_for
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

def send_registration_link(to_email, token, name):
    try:
        smtp_server = current_app.config['MAIL_SERVER']
        smtp_port = current_app.config['MAIL_PORT']
        smtp_username = current_app.config['MAIL_USERNAME']
        smtp_password = current_app.config['MAIL_PASSWORD']
        sender_email = current_app.config['MAIL_DEFAULT_SENDER']
        print(smtp_server, smtp_port, smtp_username,sender_email)
        subject = "AGRI AI : Verify Your Email"
        link = url_for('auth.confirm_email', token=token, _external=True)
        link_text = f'<a href="{link}">Click here to confirm your email</a>'
        body = f"""<p>Hi {name},</p>
        <p>Please click the link below to verify your account on AGRI-AI: {link_text}</p>
        <p>This link will expire in 1 hour.</p>
        <p>If you didn't create this account, please ignore this message.</p>"""
        
        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = to_email
        message['Subject'] = subject
        message.attach(MIMEText(body, 'html'))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            if current_app.config['MAIL_USE_TLS']:
                server.starttls()
            server.login(smtp_username, smtp_password)
            server.sendmail(sender_email, to_email, message.as_string())
        
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def send_reset_password_link(to_email, reset_url, name):
    try:
        smtp_server = current_app.config['MAIL_SERVER']
        smtp_port = current_app.config['MAIL_PORT']
        smtp_username = current_app.config['MAIL_USERNAME']
        smtp_password = current_app.config['MAIL_PASSWORD']
        sender_email = current_app.config['MAIL_DEFAULT_SENDER']

        subject = "AGRI AI : Reset Your Password"
        link_text = f'<a href="{reset_url}">Click here to reset your password</a>'
        body = f"""<p>Hi {name},</p>
        <p>Please click the link below to reset your password on AGRI-AI: {link_text}</p>
        <p>This link will expire in 1 hour.</p>
        <p>If you didn't request a password reset, please ignore this message.</p>"""
        
        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = to_email
        message['Subject'] = subject
        message.attach(MIMEText(body, 'html'))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            if current_app.config['MAIL_USE_TLS']:
                server.starttls()
            server.login(smtp_username, smtp_password)
            server.sendmail(sender_email, to_email, message.as_string())
        
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
