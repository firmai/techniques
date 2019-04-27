# Trading
The trading simulation is based on [Zipline](https://github.com/quantopian/zipline), which developed by Quantopian. It offers trading simulation based on Python.

## Setup
Install the required Zipline system packages with:
```
sudo apt-get install libatlas-base-dev python-dev gfortran pkg-config libfreetype6-dev
```


You can either wotk with Python2 or Python3. However, we also wanted to analyse our portfolio with [pyfolio](https://github.com/quantopian/pyfolio) based on an output file of Zipline. This caused some trouble around pandas versions and different versions of storing and reading .pickle files. If you want to use pyfolio, we highly recommend using Python3. Otherwise, Python2 is also an option.


### Python 2
Update Python, if your local Python version < 2.7.9 with the following steps.
First, install the required system packages.
```
sudo apt-get install -y autotools-dev blt-dev bzip2 dpkg-dev g++-multilib  gcc-multilib libbluetooth-dev libbz2-dev libexpat1-dev libffi-dev libffi6
libffi6-dbg libgdbm-dev libgpm2 libncursesw5-dev libreadline-dev libsqlite3-dev libssl-dev libtinfo-dev mime-support net-tools netbase python-crypto python-mox3  python-pil python-ply quilt tk-dev zlib1g-dev
```

Second, get Python sources and compile it.
```
wget https://www.python.org/ftp/python/2.7.11/Python-2.7.11.tgz
tar xfz Python-2.7.11.tgz
cd Python-2.7.11/
./configure --prefix /usr/local/lib/python2.7.11 --enable-ipv6
make
sudo make install
```

Rough test if the installation worked.
```
/usr/local/lib/python2.7.11/bin/python -V
Python 2.7.11
```
Go into the the trading folder of the project and create a virtual environment.
```
cd  Dinvest/trading
virtualenv --python=/usr/local/lib/python2.7.11/bin/python vzipline
```

Activate the environment.
```
source vzipline/bin/activate
```

### Python 3
Check if you have at least Python 3.4.3
```
python -V
Python 3.4.3
```

Go into the the trading folder of the project and create the virtual environment.
```
cd  Dinvest/trading
virtualenv --python=/usr/bin/python3.4 vzipline
```

Activate the environment.
```
source vzipline/bin/activate
```

### Zipline

Afterwards install Zipline via pip from your virtual environment.
```
pip install zipline
```
There is a full tutorial on the installation of Zipline also for other OS [here](http://www.zipline.io/install.html).

### pyfolio
We recommend installing pyfolio in a separate virtual environment. So just follow the steps above for setting up a virtual environment (make sure to use a different name than 'vzipline'). Install pyfolio via pip from your virtual environment.
```
pip install pyfolio
```

### Ethereum
Make sure you run a geth node or other node to interact with the blockchain. More info on this can be found at the [Ethereum wiki](https://github.com/ethereum/wiki/wiki). For a simple testcase run a geth server in the test network. If you have setup your username and password, you are all set.
```
geth --testnet --rpc
```

## TODO
- contract
  - Import exclude list from contract based on sections
  - Integrate with local running geth server account
- recommender
  - Integrate fundamentals data as pipeline
  - PE and sections need to be defined
  - Enable daily change of available capital
- trader
  - Enable daily execution in Python
- analysis
  - Create compatible pickle files for pyfolio
  - Sample analysis of buyapple
  - Sample analysis of fundamentals
