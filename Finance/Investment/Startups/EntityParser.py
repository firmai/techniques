import pickle
import sys
import json
import re
import htmlentitydefs
import codecs

class EntityParser:
	
	@staticmethod
	def LoadJsonEntity(filename):
		text = EntityParser.LoadStringEntityByFilename(filename)
		js = None
		try:
			if text:
				js = json.loads(text)
		except ValueError, e:
			print e
		return js

	@staticmethod
	def LoadStringEntityByFileHandler(fid):
		if fid:
			return pickle.load(fid)
		else:
			return ''

	@staticmethod
	def LoadStringEntityByFilename(filename, mode = 'r'):
		fid = get_file_handler(filename, mode)
		obj = None
		if fid:
			try:
				obj = pickle.load(fid)
			except ValueError, e:
				obj = None
		return obj
	
	@staticmethod
	def get_file_handler(filename, mode):
	        try:
	                fid = open(filename, mode)
	        except IOError:
	                fid = None
	        return fid
	
def main():
	js = EntityParser.LoadJsonEntity(path-to-file)

	if js:
		for i, item in enumerate(js):
			print "============="
			for key in item.keys():
				print key, "=", item[key]
	else:
		print "no json object"

if __name__ == "__main__":
	main()
