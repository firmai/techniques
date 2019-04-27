# ------------------- start ML blackbox ----------------------------
# the details here aren't fully important
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

def _split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]

pipeline_svm = Pipeline([
    ('bow', CountVectorizer(analyzer=_split_into_lemmas)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC()),  # <== change here
])

# pipeline parameters to automatically explore and tune
param_svm = [
  {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
  {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
]

def _grid_svm(label_train):
    return  GridSearchCV(
    pipeline_svm,  # pipeline from above
    param_grid=param_svm,  # parameters to tune via cross validation
    refit=True,  # fit using all data, on the best detected classifier
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
    )

# ------------------- end big ML blackbox ------------------------


from ticdat import TicDatFactory, LogFile
import cPickle
import time
import datetime


dataFactory = TicDatFactory(messages = [[],["label", "message"]],
                            parameters = [["key"], ["value"]])

solnFactory = TicDatFactory(predictions = [[],["message", "prediction"]],
                     parameters = [["key"], ["value"]])

def _timeStamp() :
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


def run(td, output, error):
    assert dataFactory.good_tic_dat_object(td)
    assert isinstance(output, LogFile) and isinstance(error, LogFile)
    output.write("Output log file for spam \n%s\n\n"%_timeStamp())
    error.write("Error log file for spam \n%s\n\n"%_timeStamp())
    messages = dataFactory.copy_to_pandas(td).messages

    # it's hard to get the type just right when reading with the csv routines.
    # to be safe, casting to strings to insure that purely numeric strings are still strings
    # this is needed for a text analysis like this example, not generally needed
    messages.message = messages.message.apply(str)

    params = dict({"mode":"predict"}, **{k:r["value"] for k,r in td.parameters.items()})
    assert params["mode"] in ["fit", "predict"]
    soln = solnFactory.TicDat()

    if params["mode"] == "fit":
        spam_detector  = _grid_svm(messages.label).fit(messages.message, messages.label)
        soln.parameters["fitted CLOB"] = cPickle.dumps(spam_detector)
        output.write("Fitted %s records\n"%len(messages))
        return soln
    else:
        if "fitted CLOB" not in params:
            error.write("Need a fitted CLOB object in order to perform a prediction.\n")
            return
        spam_detector = cPickle.loads(params["fitted CLOB"])
        rtn = spam_detector.predict(messages.message)
        assert len(rtn) == len(messages.message)
        map(soln.predictions.append, zip(messages.message, rtn))
        output.write("Predicted %s records\n"%len(messages))
        return soln
