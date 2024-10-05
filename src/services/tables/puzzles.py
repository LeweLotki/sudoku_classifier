from sqlalchemy import Column, Integer, String
from .. import Base

class Puzzle(Base):
    __tablename__ = "puzzles"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String, index=True)
    rules = Column(String, index=True)
    difficulty = Column(Integer)

    def __repr__(self):
        return f"<Puzzle(id={self.id}, code={self.code}, rules={self.rules}, difficulty={self.difficulty})>"

