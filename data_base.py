from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

# Define the database location
DATABASE_URL = "sqlite:///./species.db"

# Create the engine
engine = create_engine(DATABASE_URL, echo=True)

# Define the base class for SQLAlchemy models
Base = declarative_base()

# Define the Plants table
class Plant(Base):
    __tablename__ = "plants"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    scientific_name = Column(String)
    description = Column(Text)
    habitat = Column(String)
    flowering_season = Column(String)  # Unique to plants


# Define the Animals table
class Animal(Base):
    __tablename__ = "animals"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    scientific_name = Column(String)
    description = Column(Text)
    habitat = Column(String)
    diet = Column(String)  # Unique to animals
    

# Create the tables
#Base.metadata.create_all(bind=engine)
