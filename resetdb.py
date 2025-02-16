from app import db, create_app
from sqlalchemy import text

def clear_alembic_version():
    with create_app().app_context():
        # SQL command to clear the alembic_version table
        sql = text("DELETE FROM alembic_version;")
        db.session.execute(sql)
        db.session.commit()
        print("alembic_version table cleared.")

if __name__ == "__main__":
    clear_alembic_version()