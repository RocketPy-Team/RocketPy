#!/usr/bin/env python
import calendar
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()
 
def retrieve_interim():
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
        "date": "2013-01-01/to/2017-10-31",
        "expver": "1",
        "levtype": "pl",
        "levelist": "600/650/700/750/775/800/825/850/875/900/925/950/975/1000",
        "param": "129.128/131.128/132.128/133.128",
        "target": "CLBI.nc",
        "format": "netcdf",
        "time": "00/06/12/18",
        "grid": "0.75/0.75",
        "area": "-4.5/324/-7.5/326.25"
    })
if __name__ == '__main__':
    retrieve_interim()