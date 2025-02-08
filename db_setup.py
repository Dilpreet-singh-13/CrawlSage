# DataBase setup file, you need to only run this once to generate the schemas
# We are using supabase

from dotenv import load_dotenv
from typing import List
from datetime import datetime
import os

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, func, TIMESTAMP, create_engine, text
from sqlalchemy.schema import UniqueConstraint
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class SitePage(Base):
    __tablename__ = "site_pages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String, nullable=False)
    chunk_number: Mapped[int] = mapped_column(nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    summary: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[List[float]] = mapped_column(Vector(768), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("url", "chunk_number", name="unique_url_chunk_num"),
    )


def get_engine():
    """Makes a SQLAlchemy engine. Gets credentials from the environment variables."""
    load_dotenv()
    # Fetch variables
    USER = os.getenv("user")
    PASSWORD = os.getenv("password")
    HOST = os.getenv("host")
    PORT = os.getenv("port")
    DBNAME = os.getenv("dbname")

    DATABASE_URL = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}?sslmode=require"

    return create_engine(DATABASE_URL)


def supabase_setup():
    engine = get_engine()

    try:
        with engine.begin() as connection:
            print("Connection successful!")
            # Enable the pgvector extension
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

        # Create all tables
        Base.metadata.create_all(engine)

        with engine.begin() as connection:
            # Enable Row-Level Security (RLS)
            connection.execute(
                text("ALTER TABLE site_pages ENABLE ROW LEVEL SECURITY;")
            )
            # Create policy to allow public read access
            connection.execute(
                text("""
                CREATE POLICY "Allow public read access"
                ON site_pages
                FOR SELECT
                TO public
                USING (true);
            """)
            )
            print("RLS and public read policy activated successfully.")
    except Exception as e:
        print(f"Failed to connect: {e}")


if __name__ == "__main__":
    supabase_setup()
