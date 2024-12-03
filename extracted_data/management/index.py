import os
from dotenv import load_dotenv
from sqlalchemy import Column, Sequence, SmallInteger, Integer, Float, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool

load_dotenv()

engine = create_engine(
    os.getenv("DB_URL"),
    echo=True,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=-1,
    pool_recycle=3600,
    pool_pre_ping=True,
    connect_args={
        "connect_timeout": 60,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    },
)

Session = sessionmaker(bind=engine)

connection = engine.connect()
connection.close()

Base = declarative_base()

class PredictiveMaintenance(Base):
    __tablename__ = 'predictive_maintenance'
    
    id = Column(SmallInteger, Sequence("predictive_maintenance_id_seq"), primary_key=True)
    udi = Column('UDI', Integer)
    product_id = Column('Product ID', String)
    type = Column('Type', String)
    air_temperature = Column('Air temperature [K]', Float)
    process_temperature = Column('Process temperature [K]', Float)
    rotational_speed = Column('Rotational speed [rpm]', Integer)
    torque = Column('Torque [Nm]', Float)
    tool_wear = Column('Tool wear [min]', Integer)
    target = Column('Target', Integer)
    failure_type = Column('Failure Type', String)


     