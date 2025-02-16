"""Add project_id to result_table

Revision ID: 28377530c42a
Revises: 565c1c337db3
Create Date: 2024-10-06 15:15:02.793419

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '28377530c42a'
down_revision = '565c1c337db3'
branch_labels = None
depends_on = None


def upgrade():
    # Add project_id as a nullable column first
    with op.batch_alter_table('result_table', schema=None) as batch_op:
        batch_op.add_column(sa.Column('project_id', sa.Integer(), nullable=True))  # Initially nullable
        batch_op.create_foreign_key(None, 'project_details', ['project_id'], ['id'])

def downgrade():
    # Remove the project_id column in downgrade
    with op.batch_alter_table('result_table', schema=None) as batch_op:
        batch_op.drop_constraint(None, type_='foreignkey')
        batch_op.drop_column('project_id')



    # ### end Alembic commands ###
