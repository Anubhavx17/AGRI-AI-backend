"""Add project_id to result_table

Revision ID: 2d651d9b6d42
Revises: 28377530c42a
Create Date: 2024-10-06 16:40:23.137427

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2d651d9b6d42'
down_revision = '28377530c42a'
branch_labels = None
depends_on = None


def upgrade():
    # Add project_id as a nullable column first
    with op.batch_alter_table('result_table', schema=None) as batch_op:
        batch_op.add_column(sa.Column('project_id', sa.Integer(), nullable=True))  # Initially nullable
    
    # Set default project_id to 1 for existing rows
    op.execute('UPDATE result_table SET project_id = 1 WHERE project_id IS NULL')

    # Now make the column non-nullable
    with op.batch_alter_table('result_table', schema=None) as batch_op:
        batch_op.alter_column('project_id', nullable=False)
        batch_op.create_foreign_key(None, 'project_details', ['project_id'], ['id'])

def downgrade():
    # Downgrade to remove project_id
    with op.batch_alter_table('result_table', schema=None) as batch_op:
        batch_op.drop_constraint(None, type_='foreignkey')
        batch_op.drop_column('project_id')


    # ### end Alembic commands ###
