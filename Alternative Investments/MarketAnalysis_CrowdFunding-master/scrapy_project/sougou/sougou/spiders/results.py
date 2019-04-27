#-*- coding: utf-8 -*-
from folder import MySQL
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
    rawtable=input(print('raw table name:'))
    newtable=input(print('new table name:'))
    sql=MySQL.MySQL('root','19940228083','sougou')
    sql.create(rawtable,'(title varchar(100),pubdate varchar(100),author varchar(100),content mediumtext)')
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
        sql=MySQL.MySQL('root','19940228083','sougou')
        rawtable=ResultsSpider.rawtable
        newtable=ResultsSpider.newtable
        title = response.xpath('//h2[@class="rich_media_title"]/text()').extract_first()
        pubdate=response.xpath('//em[@id="post-date"]/text()').extract_first()
        author=response.xpath('//a[@id="post-user"]/text()').extract_first()
        content=''.join(response.xpath('//div[@class="rich_media_content "]//text()').extract()).strip()
        if title==None:
            print("WARNING:It's not a typical link, try redirect.")
            sharelink=response.xpath('//a[@id="js_share_source"]/@href').extract_first()
            yield Request(sharelink, callback=self.parse_listing)
        else:
            if pubdate==None:
                pubdate=''
            if author==None:
                author=''
            if content==None:
                content=''
            content=content.replace("'",'')
            content=content.replace('"','')
            content=content.replace("”",'')
            content=content.replace('“','')
            content=content.replace("’",'')
            content=content.replace('‘','')
            content=content.replace("\\",'')
            content=content.replace('\\','')
            title=title.replace("'",'')
            title=title.replace('"','')
            title=title.replace("“",'')
            title=title.replace('”','')
            title=title.replace("’",'')
            title=title.replace('‘','')
            title=title.replace('\\','')
            title=title.replace('\\','')
            a=0
            for j in range(len(content)):
                b=len(content)
                d=j-a
                if len(content[d].encode('utf-8'))>3:
                    content=content.replace(content[d],'')
                c=len(content)
                a=a+b-c
            sql.insert(rawtable,['title','pubdate','author','content'],[title,pubdate,author,content])
    def close(self):
        rawtable=ResultsSpider.rawtable
        newtable=ResultsSpider.newtable
        sql=MySQL.MySQL('root','19940228083','sougou')
        sql.create(newtable,' like '+rawtable)
        sql.groupby(rawtable,newtable,'title')
            