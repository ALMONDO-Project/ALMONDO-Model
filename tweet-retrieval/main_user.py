from downloadData import UserDataDownload
from downloadData import LimitsExceededError
import math
import json
import os
from datetime import datetime
from utils import *

def read_users(path):
    with open(f'{path}/users_data.json', 'r') as file:
        users_data = json.load(file)
    return users_data

def users_update(users_data, username, path):
    users_data['to_do'].remove(username)
    users_data['done'].append(username)
    with open(f'{path}/users_data.json', 'w') as file:
        json.dump(users_data, file)
    return users_data

def downloaduser(count, username): 
    try:
        user_data = UserDataDownload(username=username, count=count)
        user_data.set_limits(max_tweets_per_session=count)
        user_data.set_client()
        user_data.set_user_data()
        user_data.make_dirs()
        user_data.set_paginator(start_time=datetime(2023,1,1,0,0), end_time=datetime(2023,12,31,23,59))
        user_data.download_user_tweets()
        count = user_data.get_count()
        print(f'tweets left to download:', count)
    except ValueError as e:
        print(e)
        return
    except LimitsExceededError as e:
        print(e)
        return
    
    
         
# for username in [, , , , "@SSAB_AB", "@Roche", "@Syngenta", "@wizzair", "@Veolia", "@Siemens_Energy", "@RWE_AG", "@Vedanta_Group", "@Holcim", "@GroupeLaPoste", "@Henkel", "@Bayer", "@bp_plc", "@Michelin", "@Shell", "@TotalEnergies", "@Equinor", "@VolvoGroup", "@Airbus", "@GalpPress", "@MercedesBenz", "@BoschGlobal", "@VWGroup", "@SolvayGroup", "@covestro", "@DaimlerTruck", "@Lindeplc", "@uniper_energy", "@snam", "@eni", "@SAFRAN", "@ZF_Group", "@airliquidegroup", "@DaimlerTruck", "@Gasunie", "@NorskHydroASA", "@thyssenkrupp", "@merckgroup", "@LyondellBasell", "@AholdDelhaize", "@Naturgy", "@Severstal", "@AngloAmerican", "@ArcelorMittal", "@SkupinaCEZ", "@BATplc", "@VirginAtlantic", "@hd_materials", "@BMWGroup", "-", "@kghm_sa", "@ASMLcompany", "@InterGenEnergy", "@Ryanair", "-", "@RioTinto", "@renaultgroup", "@enagas", "@Evonik", "@BASF", "@Fresenius", "@Randstad", "@tapairportugal", "@LANXESS", "@Repsol", "@IAGAeroGroup", "@Stellantis", "@AirFranceKLM", "@FluxysGroup", "-", "@HapagLloydAG", "@Glencore", "@lufthansa", "@omv", "@cmacgm", "@Aurubis_AG", "@Lukoil", "@Grupa_PGE", "-", "-", "@SAS", "@GK_PGNiG", "@GazpromEN", "@RosneftEN", "@ECTAlliance", "@ClimateCLG", "@SolarPowerEU", "@WindEurope", "@smartEnEU", "@UNIFE", "@CER_railways", "@AVERE_EU", "@Eurelectric", "@CBItweets", "@ert_eu", "@CGF_The_Forum", "@WorldCemAssoc", "@ASDEurope", "@H2Europe", "@HydrogenCouncil", "@OEUK_", "@EuroCommerce", "@Equili_Energies", "@IETA", "@AmChamEU", "@SMMT", "@VNONCW", "@PlasticsEurope", "@Cefic", "@IGU_News", "@gd4s_eu", "@FoodDrinkEU", "@CLEPA_eu", "@ACI_EUROPE", "@ACEA_eu", "@shippingics", "@airlines_UK", "@theGCCA", "@CEMBUREAU", "@GIEBrussels", "@EU_shipping", "@Eurogas_Eu", "@eraaorg", "@CEOE_ES", "@EUROFER_eu", "@IOGP_News", "@chemieverband", "@Eurometaux", "@A4Europe", "@FuelsEurope", "@medef", "@Der_BDI", "@IFIEC_Europe", "@GasNaturallyEU", "@Confindustria", "@VDA_online", "@BusinessEurope", "@BusinessEurope", "-"]:
downloaduser(315, "@orange")
    
