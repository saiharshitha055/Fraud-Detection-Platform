import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

# Fallback to local SQLite async URL if DATABASE_URL environment variable isn't set yet
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite+aiosqlite:///./fraud_platform.db"
)

# If PostgreSQL URL is provided via environment, ensure it uses the asyncpg driver prefix
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+asyncpg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# Create the async SQLAlchemy engine
engine = create_async_engine(DATABASE_URL, echo=True, future=True)

# Create a configured "AsyncSession" class
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Base class for your database models
Base = declarative_base()

# Dependency for FastAPI endpoints to get async database sessions
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
# Function to create tables asynchronously
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)