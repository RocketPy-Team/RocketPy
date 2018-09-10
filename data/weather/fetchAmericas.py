#!/usr/bin/env python
import calendar
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()
 
def retrieve_interim():
    """      
       A function to demonstrate how to iterate efficiently over several years and months etc    
       for a particular interim_request.     
       Change the variables below to adapt the iteration to your needs.
       You can use the variable 'target' to organise the requested data in files as you wish.
       In the example below the data are organised in files per month. (eg "interim_daily_201510.grb")
    """
    yearStart = 2015
    yearEnd = 2017
    monthStart = 1
    monthEnd = 10
    for year in list(range(yearStart, yearEnd + 1)):
        for month in list(range(monthStart, monthEnd + 1)):
            startDate = '%04d%02d%02d' % (year, month, 1)
            numberOfDays = calendar.monthrange(year, month)[1]
            lastDate = '%04d%02d%02d' % (year, month, numberOfDays)
            target = "interim_daily_%04d%02d.grb" % (year, month)
            requestDates = (startDate + "/TO/" + lastDate)
            interim_request(requestDates, target)
 
def interim_request(requestDates, target):
    """      
        An ERA interim request for analysis pressure level data.
        Change the keywords below to adapt it to your needs.
        (eg to add or to remove  levels, parameters, times etc)
        Request cost per day is 112 fields, 14.2326 Mbytes
    """
    server.retrieve({
        "class": "ei",
        "stream": "oper",
        "type": "an",
        "dataset": "interim",
        "date": requestDates,
        "expver": "1",
        "levtype": "pl",
        "levelist": "600/650/700/750/775/800/825/850/875/900/925/950/975/1000",
        "param": "129.128/131.128/132.128/133.128",
        "target": target,
        "time": "00/06/12/18",
        "grid": "0.75/0.75",
        "area": "50.25/229.5/-50.25/330"
    })
if __name__ == '__main__':
    retrieve_interim()