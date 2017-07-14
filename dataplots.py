import matplotlib.pyplot as plt

#comm.BCANCER_WISCONSIN_DATASET.plot()
#plt.show()
#comm.BCANCER_WISCONSIN_DATASET.plot(kind = "bar")
#plt.show()

#print(A2.value_counts(normalize=False, sort=False))
#A2.value_counts(normalize=False, sort=False).hist()
#A2.hist(bins=50)
#plt.show()

HISTO_COLUMNS = ['A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']

def plotHisto(df):
    for column in HISTO_COLUMNS:
        df[column].hist(bins=30)
        plt.savefig('plots/'+column+'_hist.png') 
