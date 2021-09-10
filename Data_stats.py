import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from statistics import stdev, mean
import matplotlib.pyplot as plt

start_date = {}
end_date = {}
noteCount = {}

# Convert string to datetime object
def str2Date(x):
    return datetime.strptime(x, '%Y-%m-%d')

# Saves Dates
counter = 0
for gm_chunk in pd.read_csv("data/IP_progressnotes.csv", chunksize=10000, encoding='latin1'):
    print(counter)
    counter += 1
    df_relevant = gm_chunk.reset_index(drop = True)
    for i in (range(df_relevant.shape[0])):
        if (df_relevant.iloc[i]['person_id'] not in noteCount):
            noteCount[df_relevant.iloc[i]['person_id']] = 1
            start_date[df_relevant.iloc[i]['person_id']] = str2Date(df_relevant.iloc[i]['note_DATE'])
            end_date[df_relevant.iloc[i]['person_id']] = str2Date(df_relevant.iloc[i]['note_DATE'])
        else:
            noteCount[df_relevant.iloc[i]['person_id']] += 1
            if (str2Date(df_relevant.iloc[i]['note_DATE']) < start_date[df_relevant.iloc[i]['person_id']]):
                start_date[df_relevant.iloc[i]['person_id']] = str2Date(df_relevant.iloc[i]['note_DATE'])
            if (str2Date(df_relevant.iloc[i]['note_DATE']) > end_date[df_relevant.iloc[i]['person_id']]):
                end_date[df_relevant.iloc[i]['person_id']] = str2Date(df_relevant.iloc[i]['note_DATE'])


pickle.dump(start_date, open("data/dataset_stats/start_dates.p", "wb"))
pickle.dump(end_date, open("data/dataset_stats/end_dates.p", "wb"))
pickle.dump(noteCount, open("data/dataset_stats/note_counts.p", "wb"))

# Saves Notecounts
noteCount = pickle.load(open("data/dataset_stats/note_counts.p", "rb"))
start_date = pickle.load(open("data/dataset_stats/start_dates.p", "rb"))
end_date = pickle.load(open("data/dataset_stats/end_dates.p", "rb"))

counts = []
for key in noteCount:
    counts.append(noteCount[key])

print(mean(noteCount.values()))
print(stdev(noteCount.values()))

timeDiffs = []
for key in start_date:
    diff = (end_date[key] - start_date[key]).days
    if (diff > -1):
        timeDiffs.append(diff / 365.0)

# Plot
plt.rcParams.update({'font.size': 16})
cm = plt.cm.get_cmap('RdYlBu_r')
Y,X = np.histogram(counts, 25)
x_span = X.max()-X.min()
C = [cm(((x-X.min())/x_span)) for x in X]
plt.bar(X[:-1],Y,color=C,width=X[1]-X[0])
plt.yscale('Log')
plt.ylabel('# of Patients (log scaled)')
plt.xlabel('Number of Visits')
plt.title('fdfad')
plt.show()
