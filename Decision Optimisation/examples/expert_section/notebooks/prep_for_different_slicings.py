import pandas as pd
import ticdat
from itertools import product
from gurobipy import tuplelist

class DumbTupleList(object) :
    """
    imlemepents the select function in a dumb, O(n), fashion
    """
    def __init__(self, tuples):
        self._tuplelist = tuple(map(tuple, tuples))
        assert self._tuplelist, "empty tuples not helpful here"
        assert min(map(len, self._tuplelist)) == max(map(len, self._tuplelist)), "inconsistent list"
    def select(self, *args):
        assert len(args) == len(self._tuplelist[0])
        rtn = []
        for tuple in self._tuplelist:
            if all(a == "*" or a == t for a,t in zip(args, tuple)):
                rtn.append(tuple)
        return rtn

def verify(b) :
    if not b :
        raise Exception("failed!!!")

def populateTd(td, p1Len, p2Len):
    for i in range(p1Len):
        td.p1[i] = {}

    for j in range(p2Len):
        td.p2[j] = {}

    rtn = [0, 0]
    for i,j,k in product(td.p1, td.p1, td.p2):
        if i % 3 and ((i+j)%5 == 0) and (j+k)%2 :
            td.childTable[i,j,k] = {}
            if k ==3:
                rtn[0]+=1
            if j == 3 and k == 2:
                rtn[1] += 1
    ticdat.freeze_me(td)
    return rtn

def checkChildDfLen(df, chk1, chk2):
    assert chk1 and chk2
    verify(len(df.dummy.sloc[:,:,3]) == chk1)
    verify(len(df.dummy.sloc[:,3,2]) == chk2)
    verify(len(df.dummy.sloc[:,3,1]) == 0)
    verify(len(df.dummy.sloc[:,3,3]) == 0)

def checkChildDfSum(df, chk1, chk2):
    assert chk1 and chk2
    verify(df.dummy.sloc[:,:,3].sum() == chk1*2.)
    verify(df.dummy.sloc[:,3,2].sum() == chk2*2.)
    verify(df.dummy.sloc[:,3,1].sum() == 0)
    verify(df.dummy.sloc[:,3,3].sum() == 0)


def checkTupleListLen(tl, chk1, chk2):
    assert chk1 and chk2
    verify(len(tl.select("*", "*", 3)) == chk1)
    verify(len(tl.select("*", 3, 2)) == chk2)
    verify(len(tl.select("*", 3, 1)) == 0)
    verify(len(tl.select("*", 3, 3)) == 0)

def checkTupleListSum(tl, td, chk1, chk2):
    assert chk1 and chk2
    verify(sum(td.childTable[k]["dummy"] for k in tl.select("*", "*", 3)) == chk1*2.)
    verify(sum(td.childTable[k]["dummy"] for k in tl.select("*", 3, 2)) == chk2*2.)
    verify(sum(td.childTable[k]["dummy"] for k in tl.select("*", 3, 1)) == 0)
    verify(sum(td.childTable[k]["dummy"] for k in tl.select("*", 3, 3)) == 0)

def checkSlicerLen(tl, chk1, chk2):
    tl._forceguout() # for testing purposes, using underscore is ok
    assert chk1 and chk2
    verify(len(tl.slice("*", "*", 3)) == chk1)
    verify(len(tl.slice("*", 3, 2)) == chk2)
    verify(len(tl.slice("*", 3, 1)) == 0)
    verify(len(tl.slice("*", 3, 3)) == 0)
    assert tl._archived_slicings

def checkSlicerSum(tl, td, chk1, chk2):
    tl._forceguout() # for testing purposes, using underscore is ok
    assert chk1 and chk2
    verify(sum(td.childTable[k]["dummy"] for k in tl.slice("*", "*", 3)) == chk1*2.)
    verify(sum(td.childTable[k]["dummy"] for k in tl.slice("*", 3, 2)) == chk2*2.)
    verify(sum(td.childTable[k]["dummy"] for k in tl.slice("*", 3, 1)) == 0)
    verify(sum(td.childTable[k]["dummy"] for k in tl.slice("*", 3, 3)) == 0)
    assert tl._archived_slicings

# make a simple schema
tdf = ticdat.TicDatFactory(p1 = [["id"],[]], p2 = [["id"],[]], childTable = [["p1_1", "p1_2", "p2"],["dummy"]])
tdf.set_default_value("childTable", "dummy", 2.)

smallTd = tdf.TicDat()
smallChk =  populateTd(smallTd, 30, 20)
smallSmartTupleList = tuplelist(smallTd.childTable)
smallDumbTupleList = DumbTupleList(smallTd.childTable)
smallSlicer = ticdat.Slicer(smallTd.childTable)
smallChildDf = tdf.copy_to_pandas(smallTd,["childTable"]).childTable
checkChildDfLen(smallChildDf, *smallChk)
checkTupleListLen(smallSmartTupleList, *smallChk)
checkTupleListLen(smallDumbTupleList, *smallChk)
checkSlicerLen(smallSlicer, *smallChk)
checkChildDfSum(smallChildDf, *smallChk)
checkTupleListSum(smallSmartTupleList, smallTd, *smallChk)
checkTupleListSum(smallDumbTupleList, smallTd, *smallChk)
checkSlicerSum(smallSlicer, smallTd, *smallChk)

medTd = tdf.TicDat()
medChk =  populateTd(medTd, 80, 75)
medSmartTupleList = tuplelist(medTd.childTable)
medDumbTupleList = DumbTupleList(medTd.childTable)
medChildDf = tdf.copy_to_pandas(medTd,["childTable"]).childTable
medSlicer = ticdat.Slicer(medTd.childTable)
checkChildDfLen(medChildDf, *medChk)
checkTupleListLen(medSmartTupleList, *medChk)
checkTupleListLen(medDumbTupleList, *medChk)
checkSlicerLen(medSlicer, *medChk)
checkChildDfSum(medChildDf, *medChk)
checkTupleListSum(medSmartTupleList, medTd, *medChk)
checkTupleListSum(medDumbTupleList, medTd, *medChk)
checkSlicerSum(medSlicer, medTd, *medChk)

bigTd = tdf.TicDat()
bigChk =  populateTd(bigTd, 180, 125)
bigSmartTupleList = tuplelist(bigTd.childTable)
bigDumbTupleList = DumbTupleList(bigTd.childTable)
bigChildDf = tdf.copy_to_pandas(bigTd,["childTable"]).childTable
bigSlicer = ticdat.Slicer(bigTd.childTable)
checkChildDfLen(bigChildDf, *bigChk)
checkTupleListLen(bigSmartTupleList, *bigChk)
checkTupleListLen(bigDumbTupleList, *bigChk)
checkSlicerLen(bigSlicer, *bigChk)
checkChildDfSum(bigChildDf, *bigChk)
checkTupleListSum(bigSmartTupleList, bigTd, *bigChk)
checkTupleListSum(bigDumbTupleList, bigTd, *bigChk)
checkSlicerSum(bigSlicer, bigTd, *bigChk)







