import src
import logging
from os.path import join
import numpy as np
from src.tools import is_sequence
try:
    import jpype as jp
except:
    logging.warning("jpype not imported.")


def get_jar_location():

    jar_file_name = "infodynamics.jar"
    jar_location = src.__file__
    jar_location = jar_location.replace('__init__.py', '')
    jar_location = join(jar_location, jar_file_name)

    return jar_location


def init_jvm():

    jar_location = get_jar_location()

    if jp.isJVMStarted():
        return
    else:
        jp.startJVM(jp.getDefaultJVMPath(), "-ea",
                    "-Djava.class.path=" + jar_location)


def calc_TE(source, target, num_threads=1):

    init_jvm()
    calcTEClass = jp.JPackage(
        "infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    calcTE = calcTEClass()
    calcTE.setProperty("NUM_THREADS", str(num_threads))
    calcTE.initialise()
    calcTE.setObservations(source, target)
    te = calcTE.computeAverageLocalOfObservations()

    return te


def calc_MI(source, target, NUM_THREADS=1, k=4, TIME_DIFF=1):

    assert((len(source) > 0) and (len(target) > 0))
    init_jvm()

    calcClass = jp.JPackage(
        "infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov2
    calc = calcClass()
    calc.setProperty("k", str(int(k)))
    calc.setProperty("NUM_THREADS", str(int(NUM_THREADS)))
    calc.setProperty("TIME_DIFF", str(int(TIME_DIFF)))
    calc.initialise()
    calc.setObservations(source.tolist(), target.tolist())
    me = calc.computeAverageLocalOfObservations()

    return me * 1.4426950408889634  # np.log2(np.exp(1)) in bits


def entropy(x):

    x = x.flatten().tolist()
    '''
    x : list[float]
    '''

    init_jvm()

    # if len(x) > 50: #
    calcClass = jp.JPackage(
        "infodynamics.measures.continuous.kozachenko").EntropyCalculatorMultiVariateKozachenko
    calc = calcClass()
    calc.initialise()
    calc.setObservations(x)
    result = calc.computeAverageLocalOfObservations()

    return np.array([result * 1.4426950408889634])  # in bits
