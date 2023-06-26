import asyncio
import os
import random
from uuid import UUID, uuid4
from typing import Literal, Optional, Type, cast
from pydantic import BaseModel
from objectives import Objective

class Char(BaseModel):
    id: UUID
    full_name: str
    bio: str


    def __init__(
        self,
        full_name: str,
        bio: str,
        id: Optional[UUID] = None,
        ):
        if id is None:
            id = uuid4()

    def plan(self): #-> list[Objective]:
        return
