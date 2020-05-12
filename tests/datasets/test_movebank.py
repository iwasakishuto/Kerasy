# coding: utf-8
from kerasy.datasets.movebank import API

def test_movebank_api():
    api = API()
    allstudies = api.getStudies()

    gpsstudies = api.getStudiesBySensor(allstudies, 'GPS')
    api.prettyPrint(gpsstudies)

    individuals = api.getIndividualsByStudy(study_id=9493874)
    api.prettyPrint(individuals)

    gpsevents = api.getIndividualEvents(study_id=9493874, individual_id=11522613, sensor_type_id=653) #GPS events
    if len(gpsevents) > 0:
        api.prettyPrint(api.transformRawGPS(gpsevents))

    accevents = api.getIndividualEvents(study_id=9493874, individual_id=11522613, sensor_type_id=2365683) #ACC events
    if len(accevents) > 0:
        api.prettyPrint(api.transformRawACC(accevents))
