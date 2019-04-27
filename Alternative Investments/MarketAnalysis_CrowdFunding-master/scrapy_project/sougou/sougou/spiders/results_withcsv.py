#-*- coding: utf-8 -*-
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

class ResultsSpider(scrapy.Spider):
    name = 'results'
    allowed_domains = ["weixin.sogou.com","mp.weixin.qq.com"]
    def start_requests(self):
        self.driver = webdriver.Chrome('C://Users/USER/Desktop/sougou/chromedriver')
        self.driver.get('http://weixin.sogou.com')
        
        pauser=input(print('QR code login ready?'))
        query=input(print('Search query?'))
        page=input(print('Start from page:'))
        length=int(input(print('How many pages?(max100)')))
        prob='http://weixin.sogou.com/weixin?query='+query+'&_sug_type_=&s_from=input&_sug_=n&type=2&page='+page+'&ie=utf8'
        self.driver.get(prob)
        sel = Selector(text=self.driver.page_source)
        listings = sel.xpath('//h3/a[@target="_blank"]/@href').extract()
        for listing in listings:
            yield Request(listing, callback=self.parse_listing)
        pagecounter=1
        while pagecounter<length:
            try:
                next_page = self.driver.find_element_by_xpath('//a[@id="sogou_next"]')
                sleep(2)
                self.logger.info('Sleeping for 2 seconds.')
                next_page.click()

                sel = Selector(text=self.driver.page_source)
                listings = sel.xpath('//h3/a[@target="_blank"]/@href').extract()
                for listing in listings:
                    yield Request(listing, callback=self.parse_listing)
                pagecounter=pagecounter+1
            except NoSuchElementException:
                self.logger.info('No more pages to load?')
                manual=input(print('right?'))
                if manual=='yes':
                    self.driver.quit()
                    break
                else:
                    sel = Selector(text=self.driver.page_source)
                    listings = sel.xpath('//h3/a[@target="_blank"]/@href').extract()
                    for listing in listings:
                        yield Request(listing, callback=self.parse_listing)
                    pagecounter=pagecounter+1
        self.driver.quit()
    def parse_listing(self, response):
        title = response.xpath('//h2[@class="rich_media_title"]/text()').extract_first()
        pubdate=response.xpath('//em[@id="post-date"]/text()').extract_first()
        author=response.xpath('//a[@id="post-user"]/text()').extract_first()
        content=''.join(response.xpath('//div[@class="rich_media_content "]//text()').extract()).strip()
        if title==None:
            print("WARNING:It's not a typical link, try redirect.")
            sharelink=response.xpath('//a[@id="js_share_source"]/@href').extract_first()
            yield Request(sharelink, callback=self.parse_listing)
        else:
            yield {'title':title,'pubdate':pubdate,'author':author,'content':content}
    def close(self, reason):
        data=pd.read_csv(max(glob.iglob('*.csv'), key=os.path.getctime))
        a=0
        data = data.replace(np.nan, '', regex=True)
        for i in range(len(data)):
            data['content'][i]=data['content'][i].replace("'",'')
            data['content'][i]=data['content'][i].replace('"','')
            data['content'][i]=data['content'][i].replace("”",'')
            data['content'][i]=data['content'][i].replace('“','')
            data['content'][i]=data['content'][i].replace("’",'')
            data['content'][i]=data['content'][i].replace('‘','')
            data['content'][i]=data['content'][i].replace("\\",'')
            data['content'][i]=data['content'][i].replace('\\','')
            data['title'][i]=data['title'][i].replace("'",'')
            data['title'][i]=data['title'][i].replace('"','')
            data['title'][i]=data['title'][i].replace("“",'')
            data['title'][i]=data['title'][i].replace('”','')
            data['title'][i]=data['title'][i].replace("’",'')
            data['title'][i]=data['title'][i].replace('‘','')
            data['title'][i]=data['title'][i].replace('\\','')
            data['title'][i]=data['title'][i].replace('\\','')
            for j in range(len(data['content'][i])):
                b=len(data['content'][i])
                d=j-a
                if len(data['content'][i][d].encode('utf-8'))>3:
                    data['content'][i]=data['content'][i].replace(data['content'][i][d],'')
                c=len(data['content'][i])
                a=a+b-c
        cnx=mysql.connector.connect(user='root',password='password',database='sougou')
        cursor=cnx.cursor()
        rawtable=input(print('raw table name:'))
        newtable=input(print('new table name:'))
        addrawtable='CREATE TABLE '+rawtable+ '(title varchar(100),pubdate varchar(100),author varchar(100),content mediumtext)'
        cursor.execute(addrawtable)
        cnx.commit()
        for k in range(len(data)):
            add=("INSERT INTO "+rawtable+"(title,pubdate,author,content)"\
                 'VALUES("'+data['title'][k]+'","'+data['pubdate'][k]+'","'+data['author'][k]+'","'+data['content'][k]+'")')
            cursor.execute(add)
            cnx.commit()
        addnewtable='create table '+newtable+' like '+rawtable
        cursor.execute(addnewtable)
        cnx.commit()
        insertnewtable='insert ' +newtable+ ' select * from ' +rawtable+ ' group by title'
        cursor.execute(insertnewtable)
        cnx.commit()
        cursor.close()
        cnx.close()