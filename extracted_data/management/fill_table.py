import pandas as pd
from index import PredictiveMaintenance, Session, engine

#Ensure the engine connects to the database
try:
    with engine.connect() as connection:
        result = connection.execute("SELECT 1")
        print('Database connection is working: ', result.fetchone())
except Exception as e:
    print(f"Error connecting to database: {e}")



with Session.begin() as db:
    data = pd.read_csv("extracted_data/predictive_maintenance.csv")
    print(data.head(10))
    for index, row in data.iterrows():
        try:
            predictive_maintenance = PredictiveMaintenance(
            udi=row.iloc[0],
            product_id=row.iloc[1],
            type=row.iloc[2],
            air_temperature=row.iloc[3],
            process_temperature=row.iloc[4],
            rotational_speed=row.iloc[5],
            torque=row.iloc[6],
            tool_wear=row.iloc[7],
            target=row.iloc[8],
            failure_type=row.iloc[9]
            )
            db.add(predictive_maintenance)
        except Exception as e:
            print(f"Error processing row {index}: {e}")

    # Commit the session after all records are added
    db.commit()  # Commit once all entries are added