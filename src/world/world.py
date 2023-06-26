import asyncio
import os
from uuid import UUID, uuid4
import uuid
from pydantic import BaseModel
from ..char.char import Char
from typing import Literal, Optional


class Location(BaseModel):
    id: UUID
    name: str
    members: list[Char]

    def __init__(
        self,
        name: str,
        id: Optional[UUID] = None
        ):
        if id is None:
            id = uuid4()

class World(BaseModel):
    id: UUID
    name: str
    location: list[Location]

    def __init__(
        self, 
        id: Optional[UUID] = None
        ):
        if id is None:
            id = uuid4()
