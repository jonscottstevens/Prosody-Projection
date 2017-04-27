from __future__ import division

import pandas
from numpy import exp, sum

"""
The following code takes our spreadsheet and transforms it into a matrix or
1's and 0's, where the rows are melodies (e.g. 'Adv LHLH'), the columns are
QUD labels (e.g. '1a'), 1 means 'YES' and 0 means 'no'.
"""

def binarize(string):
    if "no" in string:
        return 0
    if "YES" in string:
        return 1
    return string

raw_data = pandas.read_csv("QUD-compatibility.csv")
data = raw_data.drop("Word with PA",1).applymap(binarize)
QUDs = ['1a','2','3a','3b','3d','4a','1aN','3aN','3bN','4aN']
data.columns = ['Tune'] + QUDs
intonations = ['LHLH', 'LHHL', 'LHLL', 'HLH', 'HHL', 'HLL', 'LLH', 'LHL' ,'LLL']
melodies = ['Adv ' + i for i in intonations] + \
    ['V ' + i for i in intonations] + \
    ['Aux ' + i for i in intonations] + \
    ['Subj ' + i for i in intonations]
data['Tune'] = melodies

"""
The compatible_questions function takes a melody and returns the set of all QUDs
for which the compatibility matrix registers '1' for that melody
"""

def compatible_questions(melody):
    query = 'Tune=="{}"'.format(melody)
    return [name for name in data.columns if (data.query(query)[name]==1).bool()]

"""
Speaker utility is simply the probability of the intended QUD being selected
randomly from compatible_questions(melody)
"""

def speaker_utility(QUD, melody):
    CQs = compatible_questions(melody)
    if QUD in CQs:
        return 1/len(CQs)
    else:
        return 0

"""
The following code just pre-calculates speaker utilities for all Q/M pairs
and caches them in memory for faster computation
"""

utility_cache ={}
for q in QUDs:
    to_cache = {}
    for m in melodies:
        to_cache[m] = speaker_utility(q,m)
    utility_cache[q] = to_cache

utility_cache = pandas.DataFrame(utility_cache)

def speaker_utility(QUD, melody):
    return utility_cache[QUD][melody]

"""
Production probability (soft-max of speaker utility) given a QUD and a
rationality parameter
"""

def production_probability(melody, givenQUD, rationality):
    numerator = exp(rationality * speaker_utility(givenQUD, melody))
    denominator = sum([exp(rationality * speaker_utility(givenQUD, m)) for m in melodies])
    return numerator / denominator

"""
The 'prior' function is simply a uniform prior over QUDs.
"""

def prior(QUD):
    return 1/len(QUDs)

"""
QUD_probability is the probability of a QUD given a melody and a rationality
parameter
"""

def QUD_probability(QUD, givenMelody, rationality):
    numerator = production_probability(givenMelody, QUD, rationality) * prior(QUD)
    denominator = sum([production_probability(givenMelody, q, rationality) * prior(q) for q in QUDs])
    return numerator / denominator

"""
projection is the probability of the QUD entailing the prejacent given
a melody and a rationality parameter, with the uniform prior
"""

def projection(givenMelody,rationality):
    return QUD_probability("4a",givenMelody,rationality)

"""
The following code calculates projection probability values
for all melodies for all integer rationality parameters from 1 to 10 and creates
spreadsheets of those predictions
"""

predictions = {}
for r in xrange(1,11):
    to_cache = {}
    for m in melodies:
        to_cache[m] = projection(m,r)
    predictions[r] = to_cache

predictions = pandas.DataFrame(predictions)

with open("predictions.csv", "w") as f:
    f.write(predictions.to_csv())
