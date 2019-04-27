import os
import csv
import glob
import MySQLdb
import scrapy
from selenium import webdriver
from scrapy.selector import Selector
from scrapy.http import Request
from time import sleep
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
from sqlalchemy import *
import numpy as np
import mysql.connector
class MySQL:
    def __init__(self,user,password,database):
        self.u=user
        self.p=password
        self.d=database
    def create(self,tablename,arg):
        cnx=mysql.connector.connect(user=self.u,password=self.p,database=self.d)
        cursor=cnx.cursor()
        addrawtable='CREATE TABLE '+tablename+arg
        cursor.execute(addrawtable)
        cnx.commit()
        cursor.close()
        cnx.close()
    def drop(self,tablename):
        cnx=mysql.connector.connect(user=self.u,password=self.p,database=self.d)
        cursor=cnx.cursor()
        droptable='DROP TABLE '+tablename
        cursor.execute(droptable)
        cnx.commit()
        cursor.close()
        cnx.close()
    def insert(self,table,category,content):
        cnx=mysql.connector.connect(user=self.u,password=self.p,database=self.d)
        cursor=cnx.cursor()
        add=("INSERT INTO "+table+'('+','.join(category)+')'\
                 'VALUES(\''+'\',\''.join(content)+'\')')
        cursor.execute(add)
        cnx.commit()
        cursor.close()
        cnx.close()
    def groupby(self,rawtable,newtable,category):
        cnx=mysql.connector.connect(user=self.u,password=self.p,database=self.d)
        cursor=cnx.cursor()
        insertnewtable='insert '+newtable+' select * from '+rawtable+' group by '+category
        cursor.execute(insertnewtable)
        cnx.commit()
        cursor.close()
        cnx.close()