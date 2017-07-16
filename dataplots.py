import matplotlib.pyplot as plt

# comm.BCANCER_WISCONSIN_DATASET.plot()
# plt.show()
# comm.BCANCER_WISCONSIN_DATASET.plot(kind = "bar")
# plt.show()

# print(A2.value_counts(normalize=False, sort=False))
# A2.value_counts(normalize=False, sort=False).hist()
# A2.hist(bins=50)
# plt.show()



def plotHisto(df, DATA_COLUMNS):
    for column in DATA_COLUMNS:
        df[column].hist(bins=30)
        plt.savefig('plots/' + column + '_hist.png') 
