

```python
from twitter_analytics import ReportDownloader


reports = ReportDownloader(
    username='snowdenwink',
    password='D13sn33uman!',
    from_date='03/28/2018',         # must be in string format 'mm/dd/yyyy' and nothing before October 2013 (twitter restriction).
    to_date='05/31/2018'
)

import csv


reports_filepath = reports.run()            # list of filepaths of downloaded csv reports

# Then you can parse the csv simply as follow
tweets = list()
for report in reports_filepath:
    with open(report, 'r') as csvfile:
        r = csv.DictReader(csvfile)
        rows = [row for row in r]
        tweets += rows

```


    ---------------------------------------------------------------------------

    WebDriverException                        Traceback (most recent call last)

    <ipython-input-1-ceaa50337460> in <module>()
         12 
         13 
    ---> 14 reports_filepath = reports.run()            # list of filepaths of downloaded csv reports
         15 
         16 # Then you can parse the csv simply as follow


    ~/anaconda/envs/py36/lib/python3.6/site-packages/twitter_analytics/downloader.py in run(self)
         76         :return: Pathname of the report.
         77         """
    ---> 78         self.login()
         79         self.go_to_analytics()
         80         self.go_to_report_page()


    ~/anaconda/envs/py36/lib/python3.6/site-packages/twitter_analytics/downloader.py in login(self)
        112         # Fills with credentials and click 'Log in'
        113         self.browser.find_element_by_xpath(
    --> 114             '//div[@class="LoginForm-input LoginForm-username"]/input[@type="text"]').send_keys(self.username)
        115         self.browser.find_element_by_xpath(
        116             '//div[@class="LoginForm-input LoginForm-password"]/input[@type="password"]').send_keys(self.password)


    ~/anaconda/envs/py36/lib/python3.6/site-packages/selenium/webdriver/remote/webelement.py in send_keys(self, *value)
        350         self._execute(Command.SEND_KEYS_TO_ELEMENT,
        351                       {'text': "".join(keys_to_typing(value)),
    --> 352                        'value': keys_to_typing(value)})
        353 
        354     # RenderedWebElement Items


    ~/anaconda/envs/py36/lib/python3.6/site-packages/selenium/webdriver/remote/webelement.py in _execute(self, command, params)
        499             params = {}
        500         params['id'] = self._id
    --> 501         return self._parent.execute(command, params)
        502 
        503     def find_element(self, by=By.ID, value=None):


    ~/anaconda/envs/py36/lib/python3.6/site-packages/selenium/webdriver/remote/webdriver.py in execute(self, driver_command, params)
        306         response = self.command_executor.execute(driver_command, params)
        307         if response:
    --> 308             self.error_handler.check_response(response)
        309             response['value'] = self._unwrap_value(
        310                 response.get('value', None))


    ~/anaconda/envs/py36/lib/python3.6/site-packages/selenium/webdriver/remote/errorhandler.py in check_response(self, response)
        192         elif exception_class == UnexpectedAlertPresentException and 'alert' in value:
        193             raise exception_class(message, screen, stacktrace, value['alert'].get('text'))
    --> 194         raise exception_class(message, screen, stacktrace)
        195 
        196     def _value_or_default(self, obj, key, default):


    WebDriverException: Message: unknown error: call function result missing 'value'
      (Session info: chrome=67.0.3396.87)
      (Driver info: chromedriver=2.33.506106 (8a06c39c4582fbfbab6966dbb1c38a9173bfb1a2),platform=Mac OS X 10.13.1 x86_64)


