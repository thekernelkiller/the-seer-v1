import uuid
from typing import List

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pkg.models.base import Base


class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tag: Mapped[str] = mapped_column(index=True, nullable=False)
    qid: Mapped[uuid.UUID] = mapped_column(ForeignKey("questions.id"))

    question: Mapped["Question"] = mapped_column(back_populates="tags")


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    text: Mapped[str] = mapped_column(nullable=False)
    tags: Mapped[List[Tag]] = relationship(back_populates="question", cascade="all, delete-orphan")