from index import PredictiveMaintenance, Session

with Session.begin() as db:
    result = db.query(PredictiveMaintenance).all()
    for row in result:
        print(row.type)