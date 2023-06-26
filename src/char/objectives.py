from uuid import UUID, uuid4
import os
from pydantic import BaseModel

class Objective(BaseModel):
    id: UUID
    importance: int
