import asyncpg
from typing import Optional, List, Dict, Any
from config import DATABASE_URL

_db_pool: Optional[asyncpg.Pool] = None

async def init_db():
    """
    Initializes the database connection pool and creates the 'events' table if it doesn't exist.
    Also handles migration for the 'camera_id' column if it's not of type INTEGER.
    """
    global _db_pool
    if _db_pool is None:
        _db_pool = await asyncpg.create_pool(DATABASE_URL, statement_cache_size=0)
    
    async with _db_pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id SERIAL PRIMARY KEY,
                camera_id INTEGER NOT NULL,
                object_name TEXT NOT NULL,
                event_type TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now()
            );
            """
        )
        try:
            await conn.execute("ALTER TABLE events ALTER COLUMN camera_id TYPE INTEGER USING camera_id::integer;")
            print("Database migration: camera_id column altered to INTEGER.")
        except Exception:
            pass

async def close_db():
    """
    Closes the database connection pool gracefully.
    """
    global _db_pool
    if _db_pool:
        await _db_pool.close()
        _db_pool = None

async def insert_event(camera_id: int, object_name: str, event_type: str):
    """
    Inserts a new event into the database.

    Args:
        camera_id (int): The ID of the camera that triggered the event.
        object_name (str): The name/path of the object (e.g., S3 object key).
        event_type (str): The type of event (e.g., 'frame' or 'video').
    """
    if _db_pool is None:
        raise RuntimeError("Database pool not initialized")
    
    async with _db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO events(camera_id, object_name, event_type)
            VALUES ($1, $2, $3)
            """,
            camera_id,
            object_name,
            event_type,
        )

async def list_events(camera_id: int) -> List[Dict[str, Any]]:
    """
    Retrieves a list of events for a specific camera, ordered by creation time descending.

    Args:
        camera_id (int): The ID of the camera to list events for.

    Returns:
        List[Dict[str, Any]]: A list of event records as dictionaries.
    """
    if _db_pool is None:
        raise RuntimeError("Database pool not initialized")
    
    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT object_name, event_type, created_at
            FROM events
            WHERE camera_id = $1
            ORDER BY created_at DESC
            """,
            camera_id,
        )
        return [dict(r) for r in rows]
